import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


class ImprovedReIDTracker:
    def __init__(self, reid_model_path, max_age=100, appearance_threshold=0.7):
        self.reid_model = self.load_reid_model(reid_model_path)
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.appearance_threshold = appearance_threshold

    def load_reid_model(self, model_path):
        """Загружает обученную ReID модель"""
        checkpoint = torch.load(model_path, map_location='cpu')
        # Здесь должна быть логика загрузки вашей модели
        return checkpoint

    def extract_appearance_features(self, person_crop):
        """Извлекает фичи внешности человека"""
        if self.reid_model is None:
            return np.random.rand(512)  # заглушка

        # Преобразуем crop к нужному размеру
        person_crop = cv2.resize(person_crop, (128, 256))
        person_crop = person_crop / 255.0
        person_crop = (person_crop - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        person_crop = np.transpose(person_crop, (2, 0, 1))
        person_crop = torch.FloatTensor(person_crop).unsqueeze(0)

        with torch.no_grad():
            features = self.reid_model(person_crop)

        return features.cpu().numpy().flatten()

    def update(self, detections, frame_idx):
        """
        Обновляет треки на основе новых детекций
        detections: список словарей с ключами 'bbox', 'confidence', 'crop'
        """
        # Предсказание для новых детекций
        new_features = []
        for det in detections:
            features = self.extract_appearance_features(det['crop'])
            new_features.append(features)

        # Матрица схожести
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track_id in enumerate(self.tracks.keys()):
            track_features = self.tracks[track_id]['features'][-1]  # последние фичи трека
            for j, new_feat in enumerate(new_features):
                similarity = cosine_similarity([track_features], [new_feat])[0][0]
                cost_matrix[i, j] = 1 - similarity  # преобразуем в cost

        # Венгерский алгоритм для ассоциации
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Обновление существующих треков
        matched_tracks = set()
        matched_detections = set()

        for i, j in zip(track_indices, detection_indices):
            track_id = list(self.tracks.keys())[i]
            cost = cost_matrix[i, j]

            if cost < (1 - self.appearance_threshold):
                # Совпадение найдено
                self.tracks[track_id]['bbox'] = detections[j]['bbox']
                self.tracks[track_id]['features'].append(new_features[j])
                self.tracks[track_id]['last_seen'] = frame_idx
                self.tracks[track_id]['confidence'] = detections[j]['confidence']

                matched_tracks.add(track_id)
                matched_detections.add(j)

        # Удаление старых треков
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if frame_idx - track['last_seen'] > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Создание новых треков для несопоставленных детекций
        for j in range(len(detections)):
            if j not in matched_detections:
                new_track_id = self.next_id
                self.next_id += 1

                self.tracks[new_track_id] = {
                    'bbox': detections[j]['bbox'],
                    'features': [new_features[j]],
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'confidence': detections[j]['confidence']
                }

        return self.tracks


# Интеграция с основным кодом
def integrate_improved_tracking():
    """
    Пример интеграции улучшенного трекера в основной код
    """
    # Инициализация улучшенного трекера
    reid_tracker = ImprovedReIDTracker("best_reid_model.pth")

    # В основном цикле обработки видео:
    def process_frame(frame, frame_idx, yolo_model):
        # Детекция YOLO
        results = yolo_model(frame)

        # Подготовка детекций для ReID трекера
        detections = []
        for box in results[0].boxes:
            if int(box.cls) == 0:  # person
                bbox = box.xyxy[0].cpu().numpy()
                crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                detections.append({
                    'bbox': bbox,
                    'confidence': float(box.conf),
                    'crop': crop
                })

        # Обновление треков с ReID
        tracks = reid_tracker.update(detections, frame_idx)

        return tracks