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
        frame_idx, global_id, action, cx, cy = item
        try:
            append_event(frame_idx, global_id, action, cx, cy)
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
    classes=[0, 6],
    persist=True,
    device=DEVICE,  # Только CUDA
    imgsz=640,      # Увеличиваем размер для лучшего качества
    conf=0.5,
    iou=0.5,
    verbose=True
)

window_name = "People tracking - GPU"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

position_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
trackid_to_global = {}
global_state = {}
next_global_id = 1

frame_idx = 0


def analyze_movement(positions):
    if len(positions) < 2:
        return "Stationary"
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


def find_matching_global_id(center, frame_idx):
    best_gid = None
    best_dist = None
    for gid, st in global_state.items():
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

            if track_id in trackid_to_global:
                global_id = trackid_to_global[track_id]
            else:
                match = find_matching_global_id(center, frame_idx)
                if match is not None:
                    global_id = match
                else:
                    global_id = next_global_id
                    next_global_id += 1

                trackid_to_global[track_id] = global_id

            position_history[global_id].append(center)
            movement_action = analyze_movement(position_history[global_id])

            bbox_h = y2 - y1
            size_action = "Close" if bbox_h > frame.shape[0] * 0.4 else "Far"

            action = f"{movement_action} ({size_action})"

            global_state[global_id] = {
                "last_pos": center,
                "last_frame": frame_idx,
                "last_action": action,
            }

            if frame_idx % 50 == 0:
                send_queue.put((frame_idx, global_id, action, cx, cy))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {global_id}: {action}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"People tracked: {len(global_state)} | GPU",
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