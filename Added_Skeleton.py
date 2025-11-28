from ultralytics import YOLO
import threading
import queue
import cv2
import collections
import math
import json
import numpy as np
from gs_writer import append_event

VIDEO_PATH = "videos/Drones.mp4"
MODEL_PATH = "yolo11x-pose.pt"
USE_CUDA = True

DEVICE = "cpu"

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

results = model.track(
    source=PROCESSED_VIDEO_PATH,
    stream=True,
    show=False,
    tracker="bytetrack.yaml",
    classes=[0],
    persist=True,
    device=DEVICE,
    imgsz=1024
)

window_name = "People tracking"
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


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return 180.0
    cos_angle = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def classify_pose_activity(keypoints_xy, keypoints_conf):
    if keypoints_xy is None or keypoints_conf is None:
        return "Unknown"

    kps = keypoints_xy
    conf = keypoints_conf

    if kps.shape[0] < 17 or conf.shape[0] < 17:
        return "Unknown"

    def v(i):
        return conf[i] > 0.4

    nose_idx = 0
    l_shoulder, r_shoulder = 5, 6
    l_elbow, r_elbow = 7, 8
    l_wrist, r_wrist = 9, 10
    l_hip, r_hip = 11, 12
    l_knee, r_knee = 13, 14
    l_ankle, r_ankle = 15, 16

    hands_up = False
    if v(l_wrist) and v(l_shoulder):
        if kps[l_wrist][1] < kps[l_shoulder][1]:
            hands_up = True
    if v(r_wrist) and v(r_shoulder):
        if kps[r_wrist][1] < kps[r_shoulder][1]:
            hands_up = True
    if hands_up:
        return "Hands up"

    knee_angles = []
    if v(l_hip) and v(l_knee) and v(l_ankle):
        knee_angles.append(compute_angle(kps[l_hip], kps[l_knee], kps[l_ankle]))
    if v(r_hip) and v(r_knee) and v(r_ankle):
        knee_angles.append(compute_angle(kps[r_hip], kps[r_knee], kps[r_ankle]))

    min_knee_angle = min(knee_angles) if knee_angles else 180.0

    torso_angle = None
    if v(l_shoulder) and v(l_hip):
        vec = kps[l_shoulder] - kps[l_hip]
        torso_angle = math.degrees(math.atan2(abs(vec[0]), abs(vec[1]) + 1e-6))
    elif v(r_shoulder) and v(r_hip):
        vec = kps[r_shoulder] - kps[r_hip]
        torso_angle = math.degrees(math.atan2(abs(vec[0]), abs(vec[1]) + 1e-6))

    if min_knee_angle < 120 and torso_angle is not None and torso_angle < 25:
        return "Sitting"
    if min_knee_angle < 120:
        return "Bending"
    if torso_angle is not None and torso_angle > 40:
        return "Bending"

    return "Unknown"


for result in results:
    frame_idx += 1
    frame = result.orig_img.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        ids = boxes.id

        if ids is None:
            cv2.putText(
                frame,
                f"People tracked: {len(global_state)}",
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
            continue

        ids = ids.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        kpts_xy = None
        kpts_conf = None
        if result.keypoints is not None:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()

        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            center = (cx, cy)

            if track_id in trackid_to_global:
                global_id = trackid_to_global[track_id]
            else:
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
                if best_gid is not None and best_dist is not None and best_dist <= MAX_REID_DISTANCE:
                    global_id = best_gid
                else:
                    global_id = next_global_id
                    next_global_id += 1
                trackid_to_global[track_id] = global_id

            position_history[global_id].append(center)
            movement_action = analyze_movement(position_history[global_id])

            pose_activity = "Unknown"
            if kpts_xy is not None and kpts_conf is not None and i < kpts_xy.shape[0]:
                pose_activity = classify_pose_activity(kpts_xy[i], kpts_conf[i])

            if pose_activity != "Unknown":
                base_action = pose_activity
            else:
                base_action = movement_action

            bbox_h = y2 - y1
            size_action = "Close" if bbox_h > frame.shape[0] * 0.4 else "Far"
            action = f"{base_action} ({size_action})"

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
        f"People tracked: {len(global_state)}",
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

print("\nГотово: обработка завершена.")
