from ultralytics import YOLO
import threading
import queue
import cv2
import collections
import math
import json
from gs_writer import append_event

VIDEO_PATH = "videos/test.mp4"
MODEL_PATH = "yolo11x.pt"
train_arrived = False

# Используем только CUDA
DEVICE = "cuda"

MAX_FRAME_GAP = 10 ** 10
MAX_REID_DISTANCE = 80

send_queue = queue.Queue()


def sender_worker():
    while True:
        item = send_queue.get()
        if item is None:
            break
        frame_idx, global_id, action, cx, cy, obj_type = item
        try:
            append_event(frame_idx, global_id, action, cx, cy, obj_type)
        except Exception as e:
            print("Ошибка отправки:", e)
        send_queue.task_done()


sender_thread = threading.Thread(target=sender_worker, daemon=True)
sender_thread.start()


def ensure_20_fps(video_path: str, target_fps: int = 20) -> str:
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
    parts = video_path.rsplit(".", 1)
    if len(parts) == 2:
        out_path = parts[0] + f"_fps{target_fps}." + parts[1]
    else:
        out_path = video_path + f"_fps{target_fps}.mp4"

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

model = YOLO(MODEL_PATH)

# Используем только CUDA
results = model.track(
    source=PROCESSED_VIDEO_PATH,
    stream=True,
    show=False,
    tracker="bytetrack.yaml",
    classes=[0, 6],  # 0=person, 6=train
    persist=True,
    device=DEVICE,  # Только CUDA
    imgsz=640,  # Увеличиваем размер для лучшего качества
    conf=0.5,
    iou=0.5,
    verbose=True
)

window_name = "People and Train tracking - GPU"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

position_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
size_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
trackid_to_global = {}
global_state = {}
next_person_id = 1
next_train_id = 1

frame_idx = 0


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
    if len(positions) < 2:
        return "Stopped"

    # Анализируем изменение размера (приближение/удаление)
    if len(sizes) >= 2:
        first_size = sizes[0]
        last_size = sizes[-1]
        size_change = last_size - first_size

        # Если размер увеличивается - приближается, уменьшается - удаляется
        if abs(size_change) > first_size * 0.1:  # Изменение больше 10%
            if size_change > 0:
                return "Arriving"
            else:
                return "Departing"

    # Анализируем движение по горизонтали/вертикали
    first_x, first_y = positions[0]
    last_x, last_y = positions[-1]
    dist = math.hypot(last_x - first_x, last_y - first_y)

    if dist < 5:  # Почти не двигается
        return "Stopped"
    else:
        # Определяем направление относительно камеры
        if len(sizes) >= 2:
            first_size = sizes[0]
            last_size = sizes[-1]
            size_change = last_size - first_size

            if size_change > first_size * 0.05:  # Увеличивается больше 5%
                return "Arriving"
            elif size_change < -first_size * 0.05:  # Уменьшается больше 5%
                return "Departing"

        # Если изменение размера незначительно, но есть движение
        return "Stopped" if dist < 15 else "Arriving"


def find_matching_global_id(center, frame_idx, obj_class):
    best_gid = None
    best_dist = None
    for gid, st in global_state.items():
        # Ищем только среди объектов того же класса
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


for result in results:
    frame_idx += 1
    frame = result.orig_img.copy()

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

            # Получаем класс объекта
            obj_class = int(box.cls.item())
            obj_type = get_object_type(obj_class)

            # Вычисляем размер объекта
            bbox_height = y2 - y1

            if track_id in trackid_to_global:
                global_id = trackid_to_global[track_id]
            else:
                match = find_matching_global_id(center, frame_idx, obj_class)
                if match is not None:
                    global_id = match
                else:
                    # Разные счетчики ID для разных типов объектов
                    if obj_class == 0:  # Person
                        global_id = f"P{next_person_id}"
                        next_person_id += 1
                    elif obj_class == 6:  # Train
                        global_id = f"T{next_train_id}"
                        next_train_id += 1
                    else:
                        global_id = f"O{next_person_id}"  # Other
                        next_person_id += 1

                trackid_to_global[track_id] = global_id

            position_history[global_id].append(center)

            # Разная логика анализа для разных типов объектов
            if obj_class == 6:  # Train
                size_history[global_id].append(bbox_height)
                action = analyze_train_movement(position_history[global_id], size_history[global_id])
                color = (0, 255, 255)  # Желтый для поездов
                label = f"Train {global_id}: {action}"
            else:  # Person и другие
                movement_action = analyze_person_movement(position_history[global_id])
                bbox_h = y2 - y1
                size_action = "Close" if bbox_h > frame.shape[0] * 0.4 else "Far"
                action = f"{movement_action} ({size_action})"
                color = (0, 255, 0)  # Зеленый для людей
                label = f"Person {global_id}: {action}"

            global_state[global_id] = {
                "last_pos": center,
                "last_frame": frame_idx,
                "last_action": action,
                "class": obj_class,
                "type": obj_type
            }

            if frame_idx % 50 == 0:
                send_queue.put((frame_idx, global_id, action, cx, cy, obj_type))

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

    # Статистика по типам объектов
    person_count = sum(1 for state in global_state.values() if state["class"] == 0)
    train_count = sum(1 for state in global_state.values() if state["class"] == 6)

    cv2.putText(
        frame,
        f"People: {person_count} | Trains: {train_count} | GPU",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow(window_name, frame_small)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

send_queue.join()
send_queue.put(None)
sender_thread.join()

print("\nГотово: обработка завершена на GPU.")