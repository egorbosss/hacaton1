import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F


class PersonReIDDataset(Dataset):
    """Dataset для ReID обучения"""

    def __init__(self, dataset_path, split='train', img_size=(256, 128)):
        self.dataset_path = dataset_path
        self.split = split
        self.img_size = img_size
        self.samples = []
        self.person_ids = []

        # Загружаем все изображения
        split_path = os.path.join(dataset_path, split)
        person_dirs = [d for d in os.listdir(split_path)
                       if os.path.isdir(os.path.join(split_path, d))]

        self.person_ids = sorted(person_dirs)
        self.id_to_label = {pid: idx for idx, pid in enumerate(self.person_ids)}

        for person_id in self.person_ids:
            person_path = os.path.join(split_path, person_id)
            images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png'))]

            for img_name in images:
                img_path = os.path.join(person_path, img_name)
                self.samples.append((img_path, self.id_to_label[person_id]))

        # Аугментации
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


class ReIDHead(nn.Module):
    """Голова для ReID поверх YOLO backbone"""

    def __init__(self, backbone_features, num_classes, embedding_dim=512):
        super(ReIDHead, self).__init__()
        self.embedding_dim = embedding_dim

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer
        self.embedding = nn.Linear(backbone_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, features, height, width)
        x = self.gap(x)  # (batch, features, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, features)

        # Embedding
        embedding = self.embedding(x)
        embedding = self.bn(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 нормализация

        # Classification
        x = self.dropout(embedding)
        logits = self.classifier(x)

        return logits, embedding


class ReIDTrainer:
    def __init__(self, dataset_path, model_path="yolo11n.pt"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 50
        self.embedding_dim = 512

        self.setup_datasets()
        self.setup_model()

    def setup_datasets(self):
        """Настройка datasets и dataloaders"""
        self.train_dataset = PersonReIDDataset(self.dataset_path, 'train')
        self.val_dataset = PersonReIDDataset(self.dataset_path, 'val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.num_classes = len(self.train_dataset.person_ids)
        print(f"Количество классов: {self.num_classes}")
        print(f"Размер train dataset: {len(self.train_dataset)}")
        print(f"Размер val dataset: {len(self.val_dataset)}")

    def setup_model(self):
        """Настройка модели YOLO с ReID головой"""
        # Загружаем YOLO модель
        self.yolo_model = YOLO(self.model_path)
        backbone = self.yolo_model.model.model[:-1]  # Берем backbone без detection head

        # Замораживаем начальные слои
        for param in list(backbone.parameters())[:-20]:  # Размораживаем последние 20 слоев
            param.requires_grad = False

        # Определяем размер фичей backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 128)
            backbone_output = backbone(dummy_input)
            backbone_features = backbone_output.shape[1]  # количество фичей

        # Добавляем ReID голову
        self.reid_head = ReIDHead(backbone_features, self.num_classes, self.embedding_dim)

        self.model = nn.Sequential(backbone, self.reid_head)
        self.model = self.model.to(self.device)

        # Оптимизатор и лосс функции
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0001
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_triplet = nn.TripletMarginLoss(margin=1.0)

    def get_triplet_batch(self, embeddings, labels):
        """Генерирует triplet батч для triplet loss"""
        anchors = []
        positives = []
        negatives = []

        for i, (embed, label) in enumerate(zip(embeddings, labels)):
            # Positive samples
            pos_indices = torch.where(labels == label)[0]
            pos_indices = pos_indices[pos_indices != i]

            if len(pos_indices) > 0:
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                positives.append(embeddings[pos_idx])

                # Negative samples
                neg_indices = torch.where(labels != label)[0]
                if len(neg_indices) > 0:
                    neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                    negatives.append(embeddings[neg_idx])

                    anchors.append(embed.unsqueeze(0))

        if len(anchors) > 0:
            anchors = torch.cat(anchors)
            positives = torch.cat(positives)
            negatives = torch.cat(negatives)
            return anchors, positives, negatives
        return None, None, None

    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_triplet_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, embeddings = self.model(images)

            # Classification loss
            cls_loss = self.criterion_cls(logits, labels)

            # Triplet loss
            anchors, positives, negatives = self.get_triplet_batch(embeddings, labels)
            triplet_loss = 0
            if anchors is not None:
                triplet_loss = self.criterion_triplet(anchors, positives, negatives)

            # Total loss
            loss = cls_loss + 0.5 * triplet_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            if anchors is not None:
                total_triplet_loss += triplet_loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)

        print(f'Epoch: {epoch} Train Loss: {avg_loss:.4f} '
              f'Cls Loss: {total_cls_loss / len(self.train_loader):.4f} '
              f'Triplet Loss: {total_triplet_loss / len(self.train_loader):.4f} '
              f'Accuracy: {accuracy:.2f}%')

        return avg_loss, accuracy

    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, embeddings = self.model(images)
                loss = self.criterion_cls(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)

        print(f'Epoch: {epoch} Val Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%')

        return avg_loss, accuracy

    def train(self):
        """Основной цикл обучения"""
        best_accuracy = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.scheduler.step()

            # Сохраняем лучшую модель
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model(f"best_reid_model.pth")
                print(f"Новая лучшая модель сохранена с точностью: {best_accuracy:.2f}%")

            print(f"Current LR: {self.scheduler.get_last_lr()[0]:.8f}")

    def save_model(self, path):
        """Сохраняет модель"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reid_head': self.reid_head.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.train_dataset.person_ids
        }, path)

        # Также сохраняем в YOLO формате
        self.yolo_model.save(path.replace('.pth', '_yolo.pt'))


# Запуск обучения
if __name__ == "__main__":
    dataset_path = "reid_dataset"  # Путь к подготовленному dataset'у

    trainer = ReIDTrainer(dataset_path, model_path="yolo11x.pt")
    trainer.train()