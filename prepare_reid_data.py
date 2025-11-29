import os
import cv2
import json
import shutil
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


class ReIDDataPreparer:
    def __init__(self, source_videos, output_dir="reid_dataset"):
        self.source_videos = source_videos
        self.output_dir = output_dir
        self.dataset_structure = {
            "train": {},
            "val": {},
            "test": {}
        }

    def extract_person_tracks(self, video_path, model):
        """Извлекает треки людей из видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return {}

        tracks = defaultdict(list)
        frame_idx = 0

        results = model.track(
            source=video_path,
            stream=True,
            tracker="bytetrack.yaml",
            classes=[0],  # только люди
            persist=True,
            conf=0.3,  # понизим порог для большего количества детекций
            iou=0.5,
            verbose=False
        )

        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    if int(box.cls) == 0:  # person class
                        track_id = int(box.id)
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox

                        # Проверяем размер bbox
                        if (y2 - y1) < 50 or (x2 - x1) < 30:  # слишком маленькие bbox
                            continue

                        # Сохраняем кадр с обрезанным человеком
                        frame = result.orig_img
                        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                        if person_crop.size > 0 and person_crop.shape[0] > 50 and person_crop.shape[1] > 30:
                            tracks[track_id].append({
                                'frame_idx': frame_idx,
                                'bbox': bbox,
                                'crop': person_crop
                            })

            frame_idx += 1

        cap.release()
        return tracks

    def save_track_images(self, tracks, video_name):
        """Сохраняет изображения треков в структуру dataset'а"""
        person_id_counter = 0

        for track_id, track_data in tracks.items():
            # Увеличим минимальное количество кадров для трека
            if len(track_data) < 15:
                continue

            # Берем каждый 3-й кадр для большего разнообразия
            sampled_frames = track_data[::3]
            if len(sampled_frames) < 5:  # минимум 5 изображений на человека
                continue

            person_id = f"{video_name}_person_{person_id_counter}"
            person_id_counter += 1

            # Простое разделение - берем первые 70% для train, следующие 15% для val, последние 15% для test
            total_frames = len(sampled_frames)
            train_end = int(total_frames * 0.7)
            val_end = train_end + int(total_frames * 0.15)

            train_frames = sampled_frames[:train_end]
            val_frames = sampled_frames[train_end:val_end]
            test_frames = sampled_frames[val_end:]

            # Проверяем что в каждом сплите есть хотя бы 1 изображение
            if len(train_frames) > 0:
                self.save_person_images(person_id, train_frames, "train")
            if len(val_frames) > 0:
                self.save_person_images(person_id, val_frames, "val")
            if len(test_frames) > 0:
                self.save_person_images(person_id, test_frames, "test")

            print(
                f"Сохранен человек {person_id}: train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}")

    def save_person_images(self, person_id, frames, split):
        """Сохраняет изображения конкретного человека"""
        person_dir = os.path.join(self.output_dir, split, person_id)
        os.makedirs(person_dir, exist_ok=True)

        for i, frame_data in enumerate(frames):
            try:
                img_path = os.path.join(person_dir, f"{i:04d}.jpg")

                # Ресайз изображения для единообразия
                crop = frame_data['crop']
                if crop.shape[0] > 256 or crop.shape[1] > 128:
                    crop = cv2.resize(crop, (128, 256))

                cv2.imwrite(img_path, crop)

                if person_id not in self.dataset_structure[split]:
                    self.dataset_structure[split][person_id] = []
                self.dataset_structure[split][person_id].append(img_path)
            except Exception as e:
                print(f"Ошибка сохранения изображения: {e}")
                continue

    def create_dataset_yaml(self):
        """Создает YAML файл для dataset'а"""
        dataset_yaml = {
            'path': os.path.abspath(self.output_dir),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': 1,  # один класс - человек
            'names': ['person']
        }

        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f"Создан dataset.yaml в: {os.path.join(self.output_dir, 'dataset.yaml')}")

    def prepare_dataset(self):
        """Основной метод подготовки dataset'а"""
        from ultralytics import YOLO

        print("Загружаем YOLO модель для трекинга...")
        model = YOLO("yolo11n.pt")

        # Создаем структуру директорий
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)

        total_tracks = 0
        total_persons = 0

        for video_path in self.source_videos:
            if not os.path.exists(video_path):
                print(f"Видео не найдено: {video_path}")
                continue

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Обрабатываем видео: {video_name}")

            tracks = self.extract_person_tracks(video_path, model)
            print(f"Найдено треков: {len(tracks)}")

            # Сохраняем только треки с достаточным количеством кадров
            valid_tracks = {tid: data for tid, data in tracks.items() if len(data) >= 15}
            print(f"Валидных треков (>=15 кадров): {len(valid_tracks)}")

            if valid_tracks:
                self.save_track_images(valid_tracks, video_name)
                total_tracks += len(valid_tracks)

                # Считаем количество сохраненных людей
                train_dir = os.path.join(self.output_dir, 'train')
                if os.path.exists(train_dir):
                    persons_in_video = len([d for d in os.listdir(train_dir)
                                            if d.startswith(video_name)])
                    total_persons += persons_in_video
                    print(f"Сохранено людей из видео {video_name}: {persons_in_video}")
            else:
                print(f"Нет валидных треков в видео {video_name}")

        self.create_dataset_yaml()

        # Статистика
        print(f"\n=== СТАТИСТИКА DATASET'а ===")
        print(f"Всего треков: {total_tracks}")
        print(f"Всего уникальных людей: {total_persons}")

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            if os.path.exists(split_dir):
                persons_count = len([d for d in os.listdir(split_dir)
                                     if os.path.isdir(os.path.join(split_dir, d))])
                images_count = sum([len(files) for r, d, files in os.walk(split_dir)])
                print(f"{split.upper()}: {persons_count} человек, {images_count} изображений")

        print(f"Dataset подготовлен в: {self.output_dir}")


# Использование
if __name__ == "__main__":
    # Укажите пути к вашим видеофайлам
    video_paths = [
        "videos/video.mp4"
    ]

    # Создаем директорию если не существует
    os.makedirs("videos", exist_ok=True)

    preparer = ReIDDataPreparer(video_paths)
    preparer.prepare_dataset()