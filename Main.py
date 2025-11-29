from ultralytics import YOLO
import cv2
import collections
import math
import torch
#for db
from sqlalchemy import Column, Integer, String, Float
import queue
import threading
from db import SessionLocal, Detection, reset_db
#------
#____Очистка таблицы при старте программы____
session = SessionLocal()
session.query(Detection).delete()
session.commit()
session.close()
print("[DB] drop OK")
#_____________________________________________


if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

VIDEO_PATH = "videos/ремонты.mov"
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

#Уменьшение FPS исходного видео                       
def ensure_20_fps(video_path: str, target_fps: int = 5) -> str:
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
    verbose=True
)

window_name = "People and Train tracking - " + DEVICE
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

position_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
size_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
trackid_to_global = {}
global_state = {}
next_person_id = 1
next_train_id = 1
main_train_id = None

frame_idx = 0

#for DB_____________________________________________________
db_queue = queue.Queue() 
def db_worker():
    """Фоновый поток: забирает данные из очереди и пишет их в БД."""
    session = SessionLocal()
    while True:
        item = db_queue.get()
        if item is None:
            break  # сигнал на завершение

        try:
            # вытащим только цифры из person_id, "P12" -> 12
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
            print(f"[DB] saved: person_id={pid}, frame={item['frame']}, action={item['action']}")
        except Exception as e:
            session.rollback()
            print("DB error:", e)
        finally:
            db_queue.task_done()

    session.close()
# запускаем поток
db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()
#__________________________________________________________

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

    if avg_step < 0.3 and abs(rel_size_change) < 0.02:
        return "Stopped"

    if rel_size_change > 0.05:
        return "Arriving"
    if rel_size_change < -0.05:
        return "Departing"

    if avg_step < 2.0:
        return "Moving slowly"
    return "Arriving"


def find_matching_global_id(center, frame_idx, obj_class):
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


for result in results:
    frame_idx += 1
    frame = result.orig_img.copy()
    frame_h = frame.shape[0]

    if main_train_id is not None:
        st = global_state.get(main_train_id)
        if st is None or frame_idx - st["last_frame"] > TRAIN_RESET_FRAMES:
            main_train_id = None

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

            if track_id in trackid_to_global:
                global_id = trackid_to_global[track_id]
            else:
                match = find_matching_global_id(center, frame_idx, obj_class)
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

            if obj_class == 6:
                if main_train_id is None:
                    main_train_id = global_id
                elif global_id != main_train_id:
                    continue

            position_history[global_id].append(center)

            if obj_class == 6:
                size_history[global_id].append(bbox_height)
                action = analyze_train_movement(position_history[global_id], size_history[global_id])
                color = (0, 255, 255)
                label = f"Train {global_id}: {action}"
            else:
                movement_action = analyze_person_movement(position_history[global_id])
                size_action = "Close" if bbox_height > frame_h * 0.4 else "Far"
                action = f"{movement_action} ({size_action})"
                color = (0, 255, 0)
                label = f"Person {global_id}: {action}"

            global_state[global_id] = {
                "last_pos": center,
                "last_frame": frame_idx,
                "last_action": action,
                "class": obj_class,
                "type": obj_type
            }

            if frame_idx % SEND_EVERY_N_FRAMES == 10:
                #_____________________________[DB] sending to queue________________
                db_queue.put({
                    "person_id": str(global_id),
                    "frame": int(frame_idx),
                    "action": action,
                    "x": float(cx),
                    "y": float(cy),
                })

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


#[DB]
db_queue.join()
db_queue.put(None)
db_thread.join()

print("\nГотово: обработка завершена")
