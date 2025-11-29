from ultralytics import YOLO
import cv2
import os
import collections
import math
import torch
from sqlalchemy import Column, Integer, String, Float
import queue
import threading
from db import SessionLocal, Detection, reset_db
# next for gui
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ____Очистка таблицы при старте программы____
session = SessionLocal()
session.query(Detection).delete()
session.commit()
session.close()
print("[DB] drop OK")
# _____________________________________________

# ---- выбор устройства ----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

VIDEO_PATH = "videos/video.mp4"
MODEL_PATH = "yolo11x.pt"
train_arrived = False

MAX_FRAME_GAP = 10 ** 10
MAX_REID_DISTANCE = 80
SEND_EVERY_N_FRAMES = 50
VID_STRIDE = 2

PERSON_CONF_THRESHOLD = 0.4
TRAIN_CONF_THRESHOLD = 0.7
MIN_TRAIN_REL_HEIGHT = 0.2
TRAIN_RESET_FRAMES = 200


# --------------------------------------------------
# Уменьшение FPS исходного видео
# --------------------------------------------------
def ensure_20_fps(video_path: str, target_fps: int = 5) -> str:
    # базовое имя + расширение
    base, ext = os.path.splitext(video_path)
    if ext == "":
        ext = ".mp4"

    # имя файла с нужным fps, например video_fps15.mp4
    out_path = f"{base}_fps{target_fps}{ext}"

    # 1) если уже есть перерасчитанное видео — просто используем его
    if os.path.exists(out_path):
        print(f"[FPS] найден уже пережатый файл: {out_path}, повторное сжатие не требуется")
        return out_path

    # 2) иначе — как раньше, считаем fps и при необходимости пережимаем
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео:", video_path)
        return video_path

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if orig_fps <= 0:
        cap.release()
        return video_path

    if orig_fps <= target_fps:
        cap.release()
        print(f"FPS видео {orig_fps:.2f} <= {target_fps}, перерасчет не нужен")
        return video_path

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, target_fps, (width, height))

    ratio = orig_fps / target_fps
    acc = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        acc += 1.0
        if acc >= ratio:
            out.write(frame)
            acc -= ratio

    cap.release()
    out.release()

    print(f"Видео снижено с {orig_fps:.2f} до {target_fps} fps: {out_path}")
    return out_path


PROCESSED_VIDEO_PATH = ensure_20_fps(VIDEO_PATH)

# ---- модель YOLO ----
model = YOLO(MODEL_PATH)

# --------------------------------------------------
#  DB: очередь + поток
# --------------------------------------------------
db_queue = queue.Queue()


def db_worker():
    """Фоновый поток: забирает данные из очереди и пишет их в БД."""
    session = SessionLocal()
    while True:
        item = db_queue.get()
        if item is None:
            break  # сигнал на завершение

        try:
            raw_pid = str(item["person_id"])
            pid = int(''.join(ch for ch in raw_pid if ch.isdigit()))

            det = Detection(
                person_id=pid,
                frame=item["frame"],
                action=item["action"],
                x=item["x"],
                y=item["y"],
            )
            session.add(det)
            session.commit()
            # print(f"[DB] saved: person_id={pid}, frame={item['frame']}, action={item['action']}")
        except Exception as e:
            session.rollback()
            print("DB error:", e)
        finally:
            db_queue.task_done()

    session.close()


db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()


# --------------------------------------------------
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ АНАЛИЗА
# --------------------------------------------------
def analyze_person_movement(positions):
    if len(positions) < 2:
        return "Standing"
    first_x, first_y = positions[0]
    last_x, last_y = positions[-1]
    dist = math.hypot(last_x - first_x, last_y - first_y)
    if dist < 5:
        return "Standing"
    elif dist < 20:
        return "Walking slowly"
    elif dist < 50:
        return "Walking"
    else:
        return "Moving fast"


def analyze_train_movement(positions, sizes):
    if len(positions) < 3:
        return "Stopped"

    first_x, first_y = positions[0]
    last_x, last_y = positions[-1]
    dist = math.hypot(last_x - first_x, last_y - first_y)
    steps = max(len(positions) - 1, 1)
    avg_step = dist / steps

    rel_size_change = 0.0
    if len(sizes) >= 2 and sizes[0] > 0:
        first_size = sizes[0]
        last_size = sizes[-1]
        rel_size_change = (last_size - first_size) / first_size

    # Упрощенная логика: только прибыл или уехал
    if rel_size_change > 0.05:
        return "Arrived"
    elif rel_size_change < -0.05:
        return "Departed"
    elif avg_step < 0.3 and abs(rel_size_change) < 0.02:
        return "Stopped"
    else:
        return "Stopped"


def find_matching_global_id(center, frame_idx, obj_class, global_state):
    best_gid = None
    best_dist = None
    for gid, st in global_state.items():
        if st["class"] != obj_class:
            continue
        if frame_idx - st["last_frame"] > MAX_FRAME_GAP:
            continue
        lx, ly = st["last_pos"]
        dist = math.hypot(center[0] - lx, center[1] - ly)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_gid = gid

    if best_gid is not None and best_dist <= MAX_REID_DISTANCE:
        return best_gid
    return None


def get_object_type(class_id):
    if class_id == 0:
        return "person"
    elif class_id == 6:
        return "train"
    else:
        return f"class_{class_id}"


# --------------------------------------------------
#  СИСТЕМА ЗАПОМИНАНИЯ ВРЕМЕНИ ДЕЙСТВИЙ
# --------------------------------------------------
class ActionHistory:
    def __init__(self, max_history_per_id=50):
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=max_history_per_id))
        self.last_actions = {}  # последнее действие для каждого ID
        self.arrival_times = {}  # время прибытия для поездов
        self.departure_times = {}  # время отправления для поездов

    def record_action(self, obj_id, action, timestamp):
        """Записывает действие с временной меткой"""
        # Для поездов сохраняем время прибытия и отправления
        if obj_id.startswith('T'):  # Это поезд
            if action == "Arrived" and obj_id not in self.arrival_times:
                self.arrival_times[obj_id] = timestamp
            elif action == "Departed" and obj_id not in self.departure_times:
                self.departure_times[obj_id] = timestamp

        # Если действие изменилось, записываем новую запись
        if obj_id not in self.last_actions or self.last_actions[obj_id] != action:
            self.history[obj_id].append({
                "action": action,
                "timestamp": timestamp,
                "duration": timedelta(0)  # начальная длительность
            })
            self.last_actions[obj_id] = action

        # Обновляем длительность текущего действия
        if self.history[obj_id]:
            current_record = self.history[obj_id][-1]
            if current_record["action"] == action:
                current_record["duration"] = timestamp - current_record["timestamp"]

    def get_arrival_time(self, obj_id):
        """Возвращает время прибытия поезда"""
        return self.arrival_times.get(obj_id)

    def get_departure_time(self, obj_id):
        """Возвращает время отправления поезда"""
        return self.departure_times.get(obj_id)

    def get_action_history(self, obj_id):
        """Возвращает историю действий для объекта"""
        return list(self.history[obj_id])

    def get_current_action(self, obj_id):
        """Возвращает текущее действие объекта"""
        return self.last_actions.get(obj_id, "Unknown")

    def get_current_action_with_duration(self, obj_id):
        """Возвращает текущее действие с длительностью"""
        if obj_id in self.last_actions and self.history[obj_id]:
            current = self.history[obj_id][-1]
            duration_seconds = current["duration"].total_seconds()
            return f"{current['action']} ({duration_seconds:.1f}s)"
        return "Unknown"

    def get_train_status_summary(self, obj_id):
        """Возвращает сводку статуса поезда с временем прибытия/отправления"""
        if not obj_id.startswith('T'):
            return "Not a train"

        arrival = self.get_arrival_time(obj_id)
        departure = self.get_departure_time(obj_id)
        current_action = self.get_current_action(obj_id)

        status_parts = []

        if arrival:
            status_parts.append(f"Прибыл: {arrival.strftime('%H:%M:%S')}")
        else:
            status_parts.append("Прибыл: --:--:--")

        if departure:
            status_parts.append(f"Уехал: {departure.strftime('%H:%M:%S')}")
        else:
            status_parts.append("Уехал: --:--:--")

        return " | ".join(status_parts)

    def get_recent_actions_summary(self, obj_id, max_actions=3):
        """Возвращает краткую сводку последних действий"""
        # Для поездов показываем статус с временем
        if obj_id.startswith('T'):
            return self.get_train_status_summary(obj_id)

        # Для людей - обычная история действий
        history = self.get_action_history(obj_id)
        if not history:
            return "No history"

        recent = list(history)[-max_actions:]
        summary = []
        for record in recent:
            duration_str = f"{record['duration'].total_seconds():.1f}s"
            time_str = record["timestamp"].strftime("%H:%M:%S")
            summary.append(f"{record['action']} ({duration_str}) at {time_str}")

        return " → ".join(summary)


# --------------------------------------------------
#  ФУНКЦИЯ ДАШБОРДА (YOLO + STREAMLIT)
# --------------------------------------------------
def run_dashboard():
    """
    Запускает дашборд со стримом видео и таблицами людей/поездов.
    Запуск: streamlit run Main.py
    """
    # ----- ОФОРМЛЕНИЕ СТРАНИЦЫ -----
    st.set_page_config(page_title="DFN – СИБИНТЕК", layout="wide")

    st.markdown("""
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
    .history-cell {
        max-width: 300px;
        font-size: 12px;
    }
    .train-status {
        font-size: 14px;
        font-weight: bold;
    }
    .arrived {
        color: #28a745;
    }
    .departed {
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

    # верхняя строка – время
    top_left, top_center, top_right = st.columns([2, 3, 1])
    with top_center:
        time_placeholder = st.empty()
    with top_left:
        st.image("static/logo.svg", width=200)

    # основная сетка
    video_col, people_col, train_col = st.columns([3, 2, 2])
    video_placeholder = video_col.empty()

    with people_col:
        st.markdown('<div class="block-yellow">ЛЮДИ</div>', unsafe_allow_html=True)
        people_placeholder = st.empty()

    with train_col:
        st.markdown('<div class="block-dark">ПОЕЗДА</div>', unsafe_allow_html=True)
        trains_placeholder = st.empty()

    # ----- ЛОКАЛЬНЫЕ СТРУКТУРЫ ТРЕКИНГА -----
    position_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
    size_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
    trackid_to_global = {}
    global_state = {}
    next_person_id = 1
    next_train_id = 1
    main_train_id = None
    frame_idx = 0

    # Инициализация системы запоминания действий
    action_history = ActionHistory()

    st.write("Загружаю модель YOLO…")
    # модель уже загружена глобально: model

    st.write("Запускаю трекинг видео…")

    results = model.track(
        source=PROCESSED_VIDEO_PATH,
        stream=True,
        show=False,
        tracker="bytetrack.yaml",
        classes=[0, 6],
        persist=True,
        device=DEVICE,
        imgsz=1080,
        conf=0.5,
        iou=0.5,
        vid_stride=VID_STRIDE,
        verbose=False
    )

    # ----- ГЛАВНЫЙ ЦИКЛ -----
    for result in results:
        frame_idx += 1
        frame = result.orig_img.copy()
        frame_h = frame.shape[0]

        # Текущее время для записи действий
        current_time = datetime.now()

        # время
        now = current_time.strftime("%H:%M")
        time_placeholder.markdown(f'<div class="big-time">{now}</div>', unsafe_allow_html=True)

        # сброс главного поезда
        if main_train_id is not None:
            st_train = global_state.get(main_train_id)
            if st_train is None or frame_idx - st_train["last_frame"] > TRAIN_RESET_FRAMES:
                main_train_id = None

        people_out = []
        trains_out = []

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                try:
                    track_id = int(box.id)
                except:
                    track_id = int(box.id.item())

                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                center = (cx, cy)

                obj_class = int(box.cls.item())
                obj_type = get_object_type(obj_class)
                bbox_height = y2 - y1
                conf_score = float(box.conf.item())

                # фильтры
                if obj_class == 0:
                    if conf_score < PERSON_CONF_THRESHOLD:
                        continue
                elif obj_class == 6:
                    rel_h = bbox_height / frame_h
                    if conf_score < TRAIN_CONF_THRESHOLD:
                        continue
                    if rel_h < MIN_TRAIN_REL_HEIGHT:
                        continue
                else:
                    continue

                # track_id -> global_id
                if track_id in trackid_to_global:
                    global_id = trackid_to_global[track_id]
                else:
                    match = find_matching_global_id(center, frame_idx, obj_class, global_state)
                    if match is not None:
                        global_id = match
                    else:
                        if obj_class == 0:
                            global_id = f"P{next_person_id}"
                            next_person_id += 1
                        elif obj_class == 6:
                            global_id = f"T{next_train_id}"
                            next_train_id += 1
                        else:
                            global_id = f"O{next_person_id}"
                            next_person_id += 1

                    trackid_to_global[track_id] = global_id

                # главный поезд
                if obj_class == 6:
                    if main_train_id is None:
                        main_train_id = global_id
                    elif global_id != main_train_id:
                        continue

                position_history[global_id].append(center)

                # действие
                if obj_class == 6:
                    size_history[global_id].append(bbox_height)
                    action = analyze_train_movement(position_history[global_id], size_history[global_id])
                    color = (0, 255, 255)
                    label = f"Train {global_id}: {action}"

                    # Записываем действие с временем
                    action_history.record_action(global_id, action, current_time)

                    # Получаем время прибытия и отправления
                    arrival_time = action_history.get_arrival_time(global_id)
                    departure_time = action_history.get_departure_time(global_id)

                    arrival_str = arrival_time.strftime("%H:%M:%S") if arrival_time else "--:--:--"
                    departure_str = departure_time.strftime("%H:%M:%S") if departure_time else "--:--:--"

                    trains_out.append({
                        "ID": global_id,
                        "Status": action,
                        "Arrived": arrival_str,
                        "Departed": departure_str,
                        "Current Action": action_history.get_current_action_with_duration(global_id)
                    })
                else:
                    movement_action = analyze_person_movement(position_history[global_id])
                    size_action = "Close" if bbox_height > frame_h * 0.4 else "Far"
                    action = f"{movement_action} ({size_action})"
                    color = (0, 255, 0)
                    label = f"Person {global_id}: {action}"

                    # Записываем действие с временем
                    action_history.record_action(global_id, action, current_time)

                    people_out.append({
                        "ID": global_id,
                        "Action": action_history.get_current_action_with_duration(global_id),
                        "History": action_history.get_recent_actions_summary(global_id),
                        "First Seen": action_history.get_action_history(global_id)[0]["timestamp"].strftime(
                            "%H:%M:%S") if action_history.get_action_history(global_id) else "N/A",
                        "Frame": frame_idx
                    })

                global_state[global_id] = {
                    "last_pos": center,
                    "last_frame": frame_idx,
                    "last_action": action,
                    "class": obj_class,
                    "type": obj_type
                }

                # отправка в БД раз в N кадров (люди и поезда)
                if frame_idx % SEND_EVERY_N_FRAMES == 10:
                    db_queue.put({
                        "person_id": str(global_id),
                        "frame": int(frame_idx),
                        "action": action,
                        "x": float(cx),
                        "y": float(cy),
                    })

                # рисуем bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # показываем кадр в Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)

        # таблица людей
        if people_out:
            df_people = pd.DataFrame(people_out)
            # Стилизация для лучшего отображения истории
            styled_people = df_people.style.set_properties(**{
                'max-width': '300px',
                'font-size': '12px'
            }, subset=['History'])
            people_placeholder.dataframe(styled_people, hide_index=True, use_container_width=True)
        else:
            people_placeholder.write("Нет активных людей")

        # таблица поездов
        if trains_out:
            df_trains = pd.DataFrame(trains_out)

            # Стилизация для статусов поездов
            def color_train_status(val):
                if val == "Arrived":
                    return 'color: #28a745; font-weight: bold;'
                elif val == "Departed":
                    return 'color: #dc3545; font-weight: bold;'
                else:
                    return ''

            styled_trains = df_trains.style.map(color_train_status, subset=['Status'])
            trains_placeholder.dataframe(styled_trains, hide_index=True, use_container_width=True)
        else:
            trains_placeholder.write("Нет активных поездов")

    # когда видео закончится
    st.write("Видео закончилось, обработка завершена.")

    # Вывод полной истории действий перед завершением
    st.subheader("Полная история поездов")
    all_train_ids = [obj_id for obj_id in action_history.history.keys() if obj_id.startswith('T')]
    for train_id in all_train_ids:
        arrival = action_history.get_arrival_time(train_id)
        departure = action_history.get_departure_time(train_id)

        st.write(f"**{train_id}**:")
        if arrival:
            st.write(f"  - Прибыл: {arrival.strftime('%H:%M:%S')}")
        if departure:
            st.write(f"  - Уехал: {departure.strftime('%H:%M:%S')}")

    db_queue.join()
    db_queue.put(None)
    db_thread.join()


# --------------------------------------------------
#   ТОЧКА ВХОДА
# --------------------------------------------------
if __name__ == "__main__":
    # запускать через:  streamlit run Main.py
    run_dashboard()