from ultralytics import YOLO
import cv2
import collections
import math
import torch
import queue
import threading
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import re

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except ImportError:
    pytesseract = None
    TesseractNotFoundError = Exception

from db import SessionLocal, Detection


def get_device():
    """
    Определяет устройство для вычислений.

    Returns:
        str: "cuda", "mps" или "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Устройство для инференса моделей YOLO: "cuda" / "mps" / "cpu"
DEVICE = get_device()

# Путь к входному видео для анализа
VIDEO_PATH = "videos/video.mp4"

# Модель детекции (люди, поезда)
DET_MODEL_PATH = "yolo11x.pt"

# Модель позы (keypoints людей)
POSE_MODEL_PATH = "yolo11x-pose.pt"

# При старте очищаем таблицу Detection в БД
with SessionLocal() as s:
    s.query(Detection).delete()
    s.commit()

# Эталонная высота кадра, используется для относительных порогов
FRAME_HEIGHT = 720

# Размер входного изображения для YOLO (квадрат), баланс скорость/качество
YOLO_IMGSZ = 680

# Обработка видео с пропуском кадров:
# 1 - каждый кадр, 2 - каждый второй кадр и тд
YOLO_VID_STRIDE = 3

# FPS по умолчанию, если из видео FPS не считался
DEFAULT_VIDEO_FPS = 20

# Длина истории позиций объектов для оценки движения
POSITION_HISTORY_LEN = 30

# Длина истории размеров bbox для поездов (изменение масштаба)
SIZE_HISTORY_LEN = 30

# Глобальный предел расстояния (в пикселях) для match треков (re-id)
MAX_REID_DISTANCE = 80

# Периодичность логирования в БД: каждые N кадров
SEND_EVERY_N_FRAMES = 50

# ID класса человека в модели YOLO
CLASS_PERSON = 0

# ID класса поезда в модели YOLO
CLASS_TRAIN = 6

# Базовый порог уверенности для детекции человека
PERSON_CONF_THRESHOLD = 0.25

# Более мягкий порог уверенности для людей рядом с уже найденными
PERSON_CONF_THRESHOLD_NEAR = 0.15

# Радиус (в пикселях), в котором считаем, что человек "рядом" с существующим треком
PERSON_LOCAL_RADIUS = 100.0

# Максимальный разрыв по кадрам для учета локальных людей
PERSON_LOCAL_MAX_FRAME_GAP = 10

# Максимально допустимый разрыв кадров для match треков по классам
FRAME_GAP = {
    # Люди: не матчим с треком, если его не было дольше 60 кадров
    CLASS_PERSON: 60,
    # Поезда: трек держим дольше, можно матчить после более длинного разрыва
    CLASS_TRAIN: 120,
}

# Сколько кадров объект может отсутствовать, но мы все еще его считаем "Lost"
MISSED_FRAMES = {
    # Люди: быстрее очищаем "мертвые" треки
    CLASS_PERSON: 20,
    # Поезда: допускаем больше пропусков
    CLASS_TRAIN: 60,
}

# Пороги скорости (пиксели/сек) на основе высоты кадра
# Ниже этого считаем, что человек стоит
PERSON_STANDING_SPEED = FRAME_HEIGHT * 0.02

# Между STANDING и WALK - "идет медленно"
PERSON_WALK_SLOW_SPEED = FRAME_HEIGHT * 0.08

# Между WALK SLOWLY и MOVING FAST
PERSON_WALK_SPEED = FRAME_HEIGHT * 0.2

# Порог уверенности для поезда
TRAIN_CONF_THRESHOLD = 0.7

# Минимальная относительная высота bbox поезда к кадру, фильтр мелких объектов
MIN_TRAIN_REL_HEIGHT = 0.2

# Через сколько кадров без обновления "главный поезд" сбрасывается
TRAIN_RESET_FRAMES = 200

# Коэффициенты для адаптации порога re-id относительно высоты bbox
K_REID = {
    # Для людей порог жестче
    CLASS_PERSON: 0.6,
    # Для поездов допускаем больший сдвиг
    CLASS_TRAIN: 1.2,
}

# Количество кадров, которое новое действие должно "продержаться",
# чтобы считаться стабильным и попасть в историю
ACTION_STABLE_FRAMES = 5

# Время простоя в состоянии "Standing", после которого считаем Not working
IDLE_STANDING_SEC = 3.0

# Время простоя в состоянии "Walking slowly", после которого считаем Not working
IDLE_WALK_SLOW_SEC = 5.0

# Индексы ключевых точек тела (YOLO Pose), чтобы не работать с числами напрямую
BODY_KEYPOINT_INDICES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Пары точек для отрисовки рук
SKELETON_PAIRS_ARMS = [(5, 7), (7, 9), (6, 8), (8, 10)]

# Пары точек для отрисовки ног
SKELETON_PAIRS_LEGS = [(11, 13), (13, 15), (12, 14), (14, 16)]

# Пары точек для отрисовки корпуса
SKELETON_PAIRS_BODY = [(5, 11), (6, 12)]

# Очередь для фоновой записи событий в БД
db_queue = queue.Queue()



def format_time(dt):
    """
    Форматирует datetime в строку времени.

    Args:
        dt (datetime | None): Входное время.

    Returns:
        str: Время в формате "HH:MM:SS" или "--:--:--".
    """
    if dt is None:
        return "--:--:--"
    return dt.strftime("%H:%M:%S")


def db_worker():
    """
    Фоновый воркер для записи детекций в БД.

    Читает словари из db_queue и создает записи Detection.
    Вход:
        db_queue: dict с ключами person_id, frame, action, x, y.
    Выход:
        Нет (пишет в БД).
    """
    session = SessionLocal()
    try:
        while True:
            item = db_queue.get()
            if item is None:
                break
            try:
                raw_pid = str(item["person_id"])
                pid = int("".join(ch for ch in raw_pid if ch.isdigit()))
                det = Detection(
                    person_id=pid,
                    frame=item["frame"],
                    action=item["action"],
                    x=item["x"],
                    y=item["y"],
                )
                session.add(det)
                session.commit()
            except Exception:
                session.rollback()
            finally:
                db_queue.task_done()
    finally:
        session.close()


db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()


def analyze_person_movement(positions, fps):
    """
    Анализируeт движение человека по истории координат.

    Args:
        positions (collections.deque[tuple[float, float]]):
            История центров bbox.
        fps (float): Эффективный FPS.

    Returns:
        str: Класс движения ("Standing", "Walking", ...).
    """
    if len(positions) < 2:
        return "Standing"

    pts = list(positions)
    dist = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dist += math.hypot(x1 - x0, y1 - y0)

    time_s = (len(pts) - 1) / max(fps, 1e-3)
    speed = dist / max(time_s, 1e-3)

    if speed < PERSON_STANDING_SPEED:
        return "Standing"
    if speed < PERSON_WALK_SLOW_SPEED:
        return "Walking slowly"
    if speed < PERSON_WALK_SPEED:
        return "Walking"
    return "Moving fast"


def analyze_train_movement(positions, sizes):
    """
    Анализирует состояние поезда по движению и изменению размера.

    Args:
        positions (collections.deque[tuple[float, float]]):
            История центров bbox.
        sizes (collections.deque[float]): История высоты bbox.

    Returns:
        str: "Arrived", "Departed" или "Stopped".
    """
    if len(positions) < 3:
        return "Stopped"

    x0, y0 = positions[0]
    x1, y1 = positions[-1]
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = len(positions) - 1
    avg_step = dist / max(steps, 1)

    rel_size_change = 0.0
    if len(sizes) >= 2 and sizes[0] > 0:
        s0 = sizes[0]
        s1 = sizes[-1]
        rel_size_change = (s1 - s0) / s0

    if rel_size_change > 0.05:
        return "Arrived"
    if rel_size_change < -0.05:
        return "Departed"
    if avg_step < 0.3 and abs(rel_size_change) < 0.02:
        return "Stopped"
    return "Stopped"


def get_body_kpt(kpts, confs, idx, thr=0.5):
    if kpts is None or confs is None:
        return None
    if idx >= len(kpts) or idx >= len(confs):
        return None
    if confs[idx] < thr:
        return None

    point = kpts[idx]

    # Берем только первые две координаты (x, y), даже если в точке 3+ значений
    x = float(point[0])
    y = float(point[1])

    if x == 0 or y == 0:
        return None

    return x, y



def classify_person_pose(kpts, confs, frame_h):
    """
    Классифицирует позу человека по keypoints.

    Args:
        kpts (np.ndarray | None): Координаты keypoints.
        confs (np.ndarray | None): Уверенности keypoints.
        frame_h (int): Высота кадра.

    Returns:
        str: Текстовое описание позы ("Pose: arms up", ...).
    """
    if kpts is None or confs is None:
        return "Pose: unknown"

    ls = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["left_shoulder"])
    rs = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["right_shoulder"])
    lh = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["left_hip"])
    rh = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["right_hip"])
    lw = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["left_wrist"])
    rw = get_body_kpt(kpts, confs, BODY_KEYPOINT_INDICES["right_wrist"])

    if ls and rs:
        shoulders_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    else:
        shoulders_mid = None

    if lh and rh:
        hips_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    else:
        hips_mid = None

    arms_up = False
    if ls and lw and lw[1] < ls[1] - 0.05 * frame_h:
        arms_up = True
    if rs and rw and rw[1] < rs[1] - 0.05 * frame_h:
        arms_up = True

    if shoulders_mid and hips_mid:
        dx = shoulders_mid[0] - hips_mid[0]
        dy = shoulders_mid[1] - hips_mid[1]
        if dy == 0:
            torso_angle = 90.0
        else:
            torso_angle = abs(math.degrees(math.atan2(dx, dy)))

        if arms_up:
            return "Pose: arms up"
        if torso_angle < 15:
            return "Pose: standing straight"
        if torso_angle < 40:
            return "Pose: leaning"
        return "Pose: bending"

    if arms_up:
        return "Pose: arms up"
    return "Pose: unknown"


def draw_person_skeleton(frame, kpts, confs):
    """
    Рисует скелет человека на кадре.

    Args:
        frame (np.ndarray): BGR изображение.
        kpts (np.ndarray | None): Массив keypoints.
        confs (np.ndarray | None): Массив confidence.
    """
    if kpts is None or confs is None:
        return

    def valid(idx):
        if idx >= len(kpts) or idx >= len(confs):
            return False
        if confs[idx] <= 0.5:
            return False
        x, y = kpts[idx]
        return not (x == 0 or y == 0)

    pairs = SKELETON_PAIRS_ARMS + SKELETON_PAIRS_LEGS + SKELETON_PAIRS_BODY
    for start, end in pairs:
        if not valid(start) or not valid(end):
            continue
        x1, y1 = int(kpts[start][0]), int(kpts[start][1])
        x2, y2 = int(kpts[end][0]), int(kpts[end][1])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def update_work_state(
    global_id,
    movement_action,
    pose_label,
    frame_idx,
    person_idle_state,
    fps,
):
    """
    Обновляет рабочее состояние сотрудника.

    Args:
        global_id (str): ID объекта (P...).
        movement_action (str): Класс движения.
        pose_label (str): Класс позы.
        frame_idx (int): Текущий индекс кадра.
        person_idle_state (dict): Состояние простоя по ID.
        fps (float): FPS исходного видео.

    Returns:
        tuple[str, float]: ("Working"/"Not working", время простоя в сек).
    """
    pose_active = any(
        token in pose_label.lower()
        for token in ["arms up", "bending", "leaning"]
    )
    if pose_active:
        if global_id in person_idle_state:
            person_idle_state.pop(global_id, None)
        return "Working", 0.0

    idle_types = ("Standing", "Walking slowly")
    if movement_action not in idle_types:
        if global_id in person_idle_state:
            person_idle_state.pop(global_id, None)
        return "Working", 0.0

    info = person_idle_state.get(global_id)
    if info is None or info["type"] != movement_action:
        info = {"type": movement_action, "start_frame": frame_idx}
        person_idle_state[global_id] = info

    frames_idle = frame_idx - info["start_frame"]
    seconds_idle = frames_idle * YOLO_VID_STRIDE / max(fps, 1e-3)

    if movement_action == "Standing":
        if seconds_idle >= IDLE_STANDING_SEC:
            return "Not working", seconds_idle
    else:
        if seconds_idle >= IDLE_WALK_SLOW_SEC:
            return "Not working", seconds_idle

    return "Working", seconds_idle


def find_matching_global_id(center, bbox_height, frame_idx, obj_class, global_state):
    """
    Находит подходящий global_id по близости трека.

    Args:
        center (tuple[float, float]): Центр bbox.
        bbox_height (float): Высота bbox.
        frame_idx (int): Индекс кадра.
        obj_class (int): Класс объекта (person/train).
        global_state (dict): Глобальное состояние треков.

    Returns:
        str | None: Найденный global_id или None.
    """
    best_gid = None
    best_dist = None
    best_gap = None
    cx, cy = center

    max_gap = FRAME_GAP.get(obj_class, FRAME_GAP[CLASS_PERSON])
    k_reid = K_REID.get(obj_class, K_REID[CLASS_PERSON])
    max_reid_dist = min(MAX_REID_DISTANCE, k_reid * bbox_height)

    for gid, st_state in global_state.items():
        if st_state["class"] != obj_class:
            continue

        frame_gap = frame_idx - st_state["last_frame"]
        if frame_gap > max_gap or frame_gap < 0:
            continue

        lx, ly = st_state["last_pos"]
        dist = math.hypot(cx - lx, cy - ly)
        if dist > max_reid_dist:
            continue

        if best_dist is None:
            best_dist = dist
            best_gid = gid
            best_gap = frame_gap
            continue

        if dist < best_dist:
            best_dist = dist
            best_gid = gid
            best_gap = frame_gap
        else:
            if (
                abs(dist - best_dist) <= 0.1 * max(best_dist, 1e-6)
                and frame_gap < best_gap
            ):
                best_gid = gid
                best_dist = dist
                best_gap = frame_gap

    return best_gid


def get_person_conf_threshold(center, frame_idx, global_state):
    """
    Адаптивный порог confidence для людей по локальному окружению.

    Args:
        center (tuple[float, float]): Центр текущего bbox.
        frame_idx (int): Индекс кадра.
        global_state (dict): Текущее состояние глобальных треков.

    Returns:
        float: Порог уверенности.
    """
    best_dist = None
    cx, cy = center

    for st_state in global_state.values():
        if st_state["class"] != CLASS_PERSON:
            continue

        frame_gap = frame_idx - st_state["last_frame"]
        if frame_gap > PERSON_LOCAL_MAX_FRAME_GAP or frame_gap < 0:
            continue

        lx, ly = st_state["last_pos"]
        dist = math.hypot(cx - lx, cy - ly)
        if best_dist is None or dist < best_dist:
            best_dist = dist

    if best_dist is not None and best_dist <= PERSON_LOCAL_RADIUS:
        return PERSON_CONF_THRESHOLD_NEAR
    return PERSON_CONF_THRESHOLD


class ActionHistory:
    """
    Хранит историю действий объектов и считает длительность.

    Сохраняет принятые (стабильные) действия по ID объектов.
    """

    def __init__(self, max_history_per_id=50, stable_frames=ACTION_STABLE_FRAMES):
        """
        Args:
            max_history_per_id (int): Максимум записей истории на объект.
            stable_frames (int): Число кадров для подтверждения смены действия.
        """
        self.history = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history_per_id),
        )
        self.last_actions = {}
        self.arrival_times = {}
        self.departure_times = {}
        self.stable_frames = stable_frames
        self._pending = collections.defaultdict(
            lambda: {"candidate": None, "count": 0},
        )

    def _apply_train_events(self, obj_id, action, timestamp):
        """
        Обновляет arrival/departure для поездов по действию.

        Args:
            obj_id (str): ID объекта (T...).
            action (str): Действие поезда.
            timestamp (datetime): Время действия.
        """
        if not obj_id.startswith("T"):
            return

        if action.startswith("Arrived") and obj_id not in self.arrival_times:
            self.arrival_times[obj_id] = timestamp
        elif action.startswith("Departed") and obj_id not in self.departure_times:
            self.departure_times[obj_id] = timestamp

    def record_action(self, obj_id, raw_action, timestamp):
        """
        Добавляет/обновляет действие в истории объекта.

        Args:
            obj_id (str): ID объекта.
            raw_action (str): Новое действие.
            timestamp (datetime): Время действия.
        """
        current = self.last_actions.get(obj_id)
        if current is None:
            self.last_actions[obj_id] = raw_action
            self.history[obj_id].append(
                {
                    "action": raw_action,
                    "timestamp": timestamp,
                    "duration": timedelta(0),
                },
            )
            self._apply_train_events(obj_id, raw_action, timestamp)
        else:
            if raw_action == current:
                pending = self._pending[obj_id]
                pending["candidate"] = None
                pending["count"] = 0
            else:
                pending = self._pending[obj_id]
                if pending["candidate"] == raw_action:
                    pending["count"] += 1
                else:
                    pending["candidate"] = raw_action
                    pending["count"] = 1

                if pending["count"] >= self.stable_frames:
                    self.last_actions[obj_id] = raw_action
                    self.history[obj_id].append(
                        {
                            "action": raw_action,
                            "timestamp": timestamp,
                            "duration": timedelta(0),
                        },
                    )
                    self._apply_train_events(obj_id, raw_action, timestamp)
                    pending["candidate"] = None
                    pending["count"] = 0

        if obj_id in self.last_actions and self.history[obj_id]:
            accepted_action = self.last_actions[obj_id]
            current_record = self.history[obj_id][-1]
            if current_record["action"] == accepted_action:
                current_record["duration"] = (
                    timestamp - current_record["timestamp"]
                )

    def get_arrival_time(self, obj_id):
        """
        Возвращает время прибытия поезда.

        Args:
            obj_id (str): ID поезда.

        Returns:
            datetime | None: Время прибытия.
        """
        return self.arrival_times.get(obj_id)

    def get_departure_time(self, obj_id):
        """
        Возвращает время отбытия поезда.

        Args:
            obj_id (str): ID поезда.

        Returns:
            datetime | None: Время отбытия.
        """
        return self.departure_times.get(obj_id)

    def get_first_seen_time(self, obj_id):
        """
        Возвращает время первого появления объекта.

        Args:
            obj_id (str): ID объекта.

        Returns:
            datetime | None: Время первого действия.
        """
        hist = self.history.get(obj_id)
        if not hist:
            return None
        return hist[0]["timestamp"]

    def get_current_action_with_duration(self, obj_id):
        """
        Возвращает текущее действие с длительностью.

        Args:
            obj_id (str): ID объекта.

        Returns:
            str: "Action (X.Ys)" или "Unknown".
        """
        hist = self.history.get(obj_id)
        if obj_id in self.last_actions and hist:
            current = hist[-1]
            seconds = current["duration"].total_seconds()
            return f"{current['action']} ({seconds:.1f}s)"
        return "Unknown"

    def get_person_times(self, obj_id):
        """
        Считает рабочее и общее время человека.

        Args:
            obj_id (str): ID сотрудника (P...).

        Returns:
            tuple[float, float]: (working_sec, total_sec).
        """
        hist = self.history.get(obj_id)
        if not hist:
            return 0.0, 0.0

        total = 0.0
        working = 0.0
        for record in hist:
            dur = record["duration"].total_seconds()
            total += dur
            if record["action"] == "Working":
                working += dur
        return working, total


def setup_streamlit_ui():
    """
    Настраивает UI Streamlit и возвращает плейсхолдеры.

    Returns:
        tuple: (time_placeholder, video_placeholder,
                people_placeholder, trains_placeholder).
    """
    st.set_page_config(page_title="DFN - СИБИНТЕК", layout="wide")
    st.markdown(
        """
    <style>
    p {
        color: #201600;
    }
    .stMainBlockContainer {
        background-color: #ffffff;
    }
    .stMain {
        background-color: #ffffff;
    }
    .block-yellow {
        background-color: #f6ce45;
        padding: 10px;
        color: #000;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .block-dark {
        background-color: #0e0f10;
        padding: 10px;
        color: #fff;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .big-time {
        font-size: 52px;
        font-weight: 900;
        text-align: center;
        color: #0e0f10
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    top_left, top_center, _ = st.columns([2, 3, 1])
    with top_center:
        time_placeholder = st.empty()
    with top_left:
        st.image("static/logo.svg", width=200)

    video_col, people_col, train_col = st.columns([3, 2, 2])
    video_placeholder = video_col.empty()

    with people_col:
        st.markdown(
            '<div class="block-yellow">ЛЮДИ</div>',
            unsafe_allow_html=True,
        )
        people_placeholder = st.empty()

    with train_col:
        st.markdown(
            '<div class="block-dark">ПОЕЗДА</div>',
            unsafe_allow_html=True,
        )
        trains_placeholder = st.empty()

    return time_placeholder, video_placeholder, people_placeholder, trains_placeholder


def extract_time_from_text(text, last_time):
    """
    Извлекает дату и время из OCR текста.

    Args:
        text (str): Текст от OCR.
        last_time (datetime | None): Последнее валидное время.

    Returns:
        datetime | None: Распарсенное время или None.
    """
    text = text.replace("\n", " ")

    date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
    time_match = re.search(r"(\d{2}:\d{2}:\d{2})", text)

    if not time_match and last_time is None:
        return None

    if time_match:
        time_str = time_match.group(1)
    else:
        time_str = None

    if date_match:
        date_str = date_match.group(1)
    elif last_time is not None:
        date_str = last_time.strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    if time_str is None:
        return None

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt
    except ValueError:
        return None


def ocr_video_time(frame, last_time):
    """
    Пытается прочитать timestamp с кадра через OCR.

    Args:
        frame (np.ndarray): BGR кадр видео.
        last_time (datetime | None): Последнее валидное время.

    Returns:
        datetime | None: Обновленное время или None.
    """
    if pytesseract is None:
        return None

    h, w = frame.shape[:2]
    roi = frame[0:int(h * 0.12), 0:int(w * 0.55)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789:-/ "
    text1 = pytesseract.image_to_string(th1, config=config)
    text2 = pytesseract.image_to_string(th2, config=config)

    cand1 = extract_time_from_text(text1, last_time)
    cand2 = extract_time_from_text(text2, last_time)

    if cand1 and cand2:
        if last_time is None:
            return cand1
        diff1 = abs((cand1 - last_time).total_seconds())
        diff2 = abs((cand2 - last_time).total_seconds())
        return cand1 if diff1 <= diff2 else cand2
    if cand1:
        return cand1
    if cand2:
        return cand2
    return None


def run_dashboard():
    """
    Основной пайплайн: обработка видео, аналитика и дашборд.

    Использует YOLO для трекинга, pose-модель, пишет в БД
    и рендерит результаты в Streamlit.
    """
    (
        time_placeholder,
        video_placeholder,
        people_placeholder,
        trains_placeholder,
    ) = setup_streamlit_ui()

    position_history = collections.defaultdict(
        lambda: collections.deque(maxlen=POSITION_HISTORY_LEN),
    )
    size_history = collections.defaultdict(
        lambda: collections.deque(maxlen=SIZE_HISTORY_LEN),
    )
    trackid_to_global = {}
    global_state = {}
    next_person_id = 1
    next_train_id = 1
    main_train_id = None
    frame_idx = 0
    action_history = ActionHistory()
    person_idle_state = {}

    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = DEFAULT_VIDEO_FPS

    ret, first_frame = cap.read()
    cap.release()

    last_frame_time = None
    if ret:
        last_frame_time = ocr_video_time(first_frame, None)
    if last_frame_time is None:
        last_frame_time = datetime.now()

    effective_fps = video_fps / YOLO_VID_STRIDE

    det_model = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    results = det_model.track(
        source=VIDEO_PATH,
        stream=True,
        show=False,
        tracker="bytetrack.yaml",
        classes=[CLASS_PERSON, CLASS_TRAIN],
        persist=True,
        device=DEVICE,
        imgsz=YOLO_IMGSZ,
        conf=0.35,
        iou=0.2,
        vid_stride=YOLO_VID_STRIDE,
        verbose=False,
    )

    for result in results:
        frame_idx += 1
        frame = result.orig_img.copy()
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        detected_time = ocr_video_time(frame, last_frame_time)
        if detected_time is not None:
            frame_time = detected_time
        else:
            delta_s = YOLO_VID_STRIDE / max(video_fps, 1e-3)
            frame_time = last_frame_time + timedelta(seconds=delta_s)
        last_frame_time = frame_time

        time_placeholder.markdown(
            f'<div class="big-time">'
            f'{frame_time.strftime("%H:%M:%S")}'
            f"</div>",
            unsafe_allow_html=True,
        )

        if main_train_id is not None:
            st_train = global_state.get(main_train_id)
            if (
                st_train is None
                or frame_idx - st_train["last_frame"] > TRAIN_RESET_FRAMES
            ):
                main_train_id = None

        people_out = []
        trains_out = []

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                try:
                    track_id = int(box.id)
                except Exception:
                    track_id = int(box.id.item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1 = max(0, min(frame_w - 1, int(x1)))
                y1 = max(0, min(frame_h - 1, int(y1)))
                x2 = max(0, min(frame_w - 1, int(x2)))
                y2 = max(0, min(frame_h - 1, int(y2)))

                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                center = (cx, cy)

                obj_class = int(box.cls.item())
                bbox_height = float(y2 - y1)
                conf_score = float(box.conf.item())

                if obj_class == CLASS_PERSON:
                    local_thr = get_person_conf_threshold(
                        center,
                        frame_idx,
                        global_state,
                    )
                    if conf_score < local_thr:
                        continue
                elif obj_class == CLASS_TRAIN:
                    rel_h = bbox_height / frame_h
                    if (
                        conf_score < TRAIN_CONF_THRESHOLD
                        or rel_h < MIN_TRAIN_REL_HEIGHT
                    ):
                        continue
                else:
                    continue

                if track_id in trackid_to_global:
                    global_id = trackid_to_global[track_id]
                else:
                    match = find_matching_global_id(
                        center,
                        bbox_height,
                        frame_idx,
                        obj_class,
                        global_state,
                    )
                    if match is not None:
                        global_id = match
                    else:
                        if obj_class == CLASS_PERSON:
                            global_id = f"P{next_person_id}"
                            next_person_id += 1
                        elif obj_class == CLASS_TRAIN:
                            global_id = f"T{next_train_id}"
                            next_train_id += 1
                        else:
                            continue
                    trackid_to_global[track_id] = global_id

                if obj_class == CLASS_TRAIN:
                    if main_train_id is None:
                        main_train_id = global_id
                    elif global_id != main_train_id:
                        continue

                position_history[global_id].append(center)

                if obj_class == CLASS_TRAIN:
                    size_history[global_id].append(bbox_height)
                    train_action = analyze_train_movement(
                        position_history[global_id],
                        size_history[global_id],
                    )
                    color = (0, 255, 255)
                    label = f"Train {global_id}: {train_action}"
                    stored_action = train_action

                    action_history.record_action(
                        global_id,
                        stored_action,
                        frame_time,
                    )

                    arrival_str = format_time(
                        action_history.get_arrival_time(global_id),
                    )
                    departure_str = format_time(
                        action_history.get_departure_time(global_id),
                    )

                    trains_out.append(
                        {
                            "ID": global_id,
                            "Status": train_action,
                            "Arrived": arrival_str,
                            "Departed": departure_str,
                            "Current Action": (
                                action_history.get_current_action_with_duration(
                                    global_id,
                                )
                            ),
                        },
                    )
                else:
                    crop = frame[y1:y2, x1:x2]
                    pose_label = "Pose: unknown"

                    if crop.size > 0:
                        pose_res_list = pose_model(crop, imgsz=YOLO_IMGSZ, verbose=False)
                        if pose_res_list:
                            pose_res = pose_res_list[0]
                            if pose_res.keypoints is not None and len(pose_res.keypoints) > 0:
                                kpts = pose_res.keypoints.data[0].cpu().numpy()
                                if hasattr(pose_res.keypoints, "conf") and pose_res.keypoints.conf is not None:
                                    confs = pose_res.keypoints.conf[0].cpu().numpy()
                                else:
                                    confs = np.ones(kpts.shape[0], dtype=np.float32)

                                pose_label = classify_person_pose(kpts, confs, frame_h)

                                kpts_draw = kpts[:, :2] + np.array([x1, y1])
                                draw_person_skeleton(frame, kpts_draw, confs)

                    movement_action = analyze_person_movement(
                        position_history[global_id],
                        effective_fps,
                    )
                    size_action = (
                        "Close" if bbox_height > frame_h * 0.4 else "Far"
                    )

                    work_state, idle_seconds = update_work_state(
                        global_id,
                        movement_action,
                        pose_label,
                        frame_idx,
                        person_idle_state,
                        video_fps,
                    )

                    detail = (
                        f"{movement_action} ({size_action}) | "
                        f"{pose_label} | idle={idle_seconds:.1f}s"
                    )
                    color = (
                        (0, 255, 0)
                        if work_state == "Working"
                        else (0, 0, 255)
                    )
                    label = f"Person {global_id}: {work_state}"
                    stored_action = work_state

                    action_history.record_action(
                        global_id,
                        stored_action,
                        frame_time,
                    )

                    first_seen_time = action_history.get_first_seen_time(
                        global_id,
                    )
                    if first_seen_time:
                        first_seen_str = format_time(first_seen_time)
                    else:
                        first_seen_str = "N/A"

                    working_sec, total_sec = (
                        action_history.get_person_times(global_id)
                    )
                    if total_sec > 0.0:
                        kpi_value = working_sec / total_sec
                        kpi_str = f"{kpi_value:.2f}"
                    else:
                        kpi_str = "N/A"

                    people_out.append(
                        {
                            "ID": global_id,
                            "Work": work_state,
                            "Action": (
                                action_history
                                .get_current_action_with_duration(global_id)
                            ),
                            "Details": detail,
                            "First Seen": first_seen_str,
                            "Frame": frame_idx,
                            "KPI": kpi_str,
                        },
                    )

                global_state[global_id] = {
                    "last_pos": center,
                    "last_frame": frame_idx,
                    "last_action": stored_action,
                    "class": obj_class,
                    "bbox": (x1, y1, x2, y2),
                }

                if frame_idx % SEND_EVERY_N_FRAMES == 10:
                    db_queue.put(
                        {
                            "person_id": str(global_id),
                            "frame": int(frame_idx),
                            "action": stored_action,
                            "x": float(cx),
                            "y": float(cy),
                        },
                    )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        displayed_people_ids = {p["ID"] for p in people_out}
        displayed_train_ids = {t["ID"] for t in trains_out}

        for gid, state in global_state.items():
            obj_class = state["class"]
            max_missed = MISSED_FRAMES.get(
                obj_class,
                MISSED_FRAMES[CLASS_PERSON],
            )
            frame_gap = frame_idx - state["last_frame"]
            if frame_gap <= 0 or frame_gap > max_missed:
                continue

            bbox = state.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            last_action = state.get("last_action", "Unknown")
            lost_label_suffix = " (Lost)"

            if obj_class == CLASS_PERSON:
                if gid in displayed_people_ids:
                    continue

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Person {gid}: {last_action}{lost_label_suffix}",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                first_seen_time = action_history.get_first_seen_time(gid)
                if first_seen_time:
                    first_seen_str = format_time(first_seen_time)
                else:
                    first_seen_str = "N/A"

                working_sec, total_sec = (
                    action_history.get_person_times(gid)
                )
                if total_sec > 0.0:
                    kpi_value = working_sec / total_sec
                    kpi_str = f"{kpi_value:.2f}"
                else:
                    kpi_str = "N/A"

                people_out.append(
                    {
                        "ID": gid,
                        "Work": last_action,
                        "Action": (
                            action_history
                            .get_current_action_with_duration(gid)
                        ),
                        "Details": "",
                        "First Seen": first_seen_str,
                        "Frame": state["last_frame"],
                        "KPI": kpi_str,
                    },
                )
            elif obj_class == CLASS_TRAIN:
                if gid in displayed_train_ids:
                    continue
                if main_train_id is not None and gid != main_train_id:
                    continue

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Train {gid}: {last_action}{lost_label_suffix}",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                arrival_str = format_time(
                    action_history.get_arrival_time(gid),
                )
                departure_str = format_time(
                    action_history.get_departure_time(gid),
                )

                trains_out.append(
                    {
                        "ID": gid,
                        "Status": last_action,
                        "Arrived": arrival_str,
                        "Departed": departure_str,
                        "Current Action": (
                            action_history
                            .get_current_action_with_duration(gid)
                        ),
                    },
                )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)

        if people_out:
            df_people = pd.DataFrame(people_out)
            styled_people = df_people.style.set_properties(
                **{"max-width": "300px", "font-size": "12px"},
                subset=["Details"],
            )
            people_placeholder.dataframe(
                styled_people,
                hide_index=True,
                use_container_width=True,
            )
        else:
            people_placeholder.write("Нет активных людей")

        if trains_out:
            df_trains = pd.DataFrame(trains_out)

            def color_train_status(val):
                if val == "Arrived":
                    return "color: #28a745; font-weight: bold;"
                if val == "Departed":
                    return "color: #dc3545; font-weight: bold;"
                return ""

            styled_trains = df_trains.style.map(
                color_train_status,
                subset=["Status"],
            )
            trains_placeholder.dataframe(
                styled_trains,
                hide_index=True,
                use_container_width=True,
            )
        else:
            trains_placeholder.write("Нет активных поездов")

    st.subheader("Полная история поездов")
    all_train_ids = [
        obj_id
        for obj_id in action_history.history.keys()
        if obj_id.startswith("T")
    ]
    for train_id in all_train_ids:
        arrival = action_history.get_arrival_time(train_id)
        departure = action_history.get_departure_time(train_id)
        st.write(f"**{train_id}**:")
        if arrival:
            st.write(f"  - Прибыл: {format_time(arrival)}")
        if departure:
            st.write(f"  - Уехал: {format_time(departure)}")

    st.subheader("KPI сотрудников")
    person_ids = [
        obj_id
        for obj_id in action_history.history.keys()
        if obj_id.startswith("P")
    ]
    kpi_rows = []
    for pid in person_ids:
        working_sec, total_sec = action_history.get_person_times(pid)
        if total_sec <= 0.0:
            continue
        kpi_val = working_sec / total_sec
        kpi_rows.append(
            {
                "ID": pid,
                "KPI": round(kpi_val, 2),
                "Working time, s": round(working_sec, 1),
                "Total time in scene, s": round(total_sec, 1),
            },
        )

    if kpi_rows:
        df_kpi = pd.DataFrame(kpi_rows)
        st.dataframe(
            df_kpi,
            hide_index=True,
            use_container_width=True,
        )
        median_kpi = float(np.median(df_kpi["KPI"].values))
        st.metric("Медианный KPI по сотрудникам", f"{median_kpi:.2f}")
    else:
        st.write("Нет данных для расчета KPI")

    db_queue.join()
    db_queue.put(None)
    db_thread.join()


if __name__ == "__main__":
    run_dashboard()
