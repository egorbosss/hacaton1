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

from db import SessionLocal, Detection


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

VIDEO_PATH = "videos/video.mp4"
MODEL_PATH = "yolo11x.pt"

with SessionLocal() as s:
    s.query(Detection).delete()
    s.commit()

FRAME_HEIGHT = 720

YOLO_IMGSZ = 960
YOLO_VID_STRIDE = 2

VIDEO_FPS = 20
EFFECTIVE_FPS = VIDEO_FPS / YOLO_VID_STRIDE

POSITION_HISTORY_LEN = 30
SIZE_HISTORY_LEN = 30

MAX_REID_DISTANCE = 80
SEND_EVERY_N_FRAMES = 50

CLASS_PERSON = 0
CLASS_TRAIN = 6

PERSON_CONF_THRESHOLD = 0.25
PERSON_CONF_THRESHOLD_NEAR = 0.15
PERSON_LOCAL_RADIUS = 100.0
PERSON_LOCAL_MAX_FRAME_GAP = 10

FRAME_GAP = {
    CLASS_PERSON: 60,
    CLASS_TRAIN: 120,
}
MISSED_FRAMES = {
    CLASS_PERSON: 20,
    CLASS_TRAIN: 60,
}

PERSON_STANDING_SPEED = FRAME_HEIGHT * 0.02
PERSON_WALK_SLOW_SPEED = FRAME_HEIGHT * 0.08
PERSON_WALK_SPEED = FRAME_HEIGHT * 0.2

TRAIN_CONF_THRESHOLD = 0.7
MIN_TRAIN_REL_HEIGHT = 0.2
TRAIN_RESET_FRAMES = 200

K_REID = {
    CLASS_PERSON: 0.6,
    CLASS_TRAIN: 1.2,
}

ACTION_STABLE_FRAMES = 5

db_queue = queue.Queue()


def format_time(dt):
    if dt is None:
        return "--:--:--"
    return dt.strftime("%H:%M:%S")


def db_worker():
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
    if len(positions) < 2:
        return "Standing"

    pts = list(positions)
    dist = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dist += math.hypot(x1 - x0, y1 - y0)

    time_s = (len(pts) - 1) / fps
    speed = dist / max(time_s, 1e-3)

    if speed < PERSON_STANDING_SPEED:
        return "Standing"
    if speed < PERSON_WALK_SLOW_SPEED:
        return "Walking slowly"
    if speed < PERSON_WALK_SPEED:
        return "Walking"
    return "Moving fast"


def analyze_train_movement(positions, sizes):
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


def find_matching_global_id(center, bbox_height, frame_idx, obj_class, global_state):
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
            if abs(dist - best_dist) <= 0.1 * max(best_dist, 1e-6) and frame_gap < best_gap:
                best_gid = gid
                best_dist = dist
                best_gap = frame_gap

    return best_gid


def get_person_conf_threshold(center, frame_idx, global_state):
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
    def __init__(self, max_history_per_id=50, stable_frames=ACTION_STABLE_FRAMES):
        self.history = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history_per_id)
        )
        self.last_actions = {}
        self.arrival_times = {}
        self.departure_times = {}
        self.stable_frames = stable_frames
        self._pending = collections.defaultdict(
            lambda: {"candidate": None, "count": 0}
        )

    def _apply_train_events(self, obj_id, action, timestamp):
        if not obj_id.startswith("T"):
            return
        if action == "Arrived" and obj_id not in self.arrival_times:
            self.arrival_times[obj_id] = timestamp
        elif action == "Departed" and obj_id not in self.departure_times:
            self.departure_times[obj_id] = timestamp

    def record_action(self, obj_id, raw_action, timestamp):
        current = self.last_actions.get(obj_id)

        if current is None:
            self.last_actions[obj_id] = raw_action
            self.history[obj_id].append(
                {
                    "action": raw_action,
                    "timestamp": timestamp,
                    "duration": timedelta(0),
                }
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
                        }
                    )
                    self._apply_train_events(obj_id, raw_action, timestamp)
                    pending["candidate"] = None
                    pending["count"] = 0

        if obj_id in self.last_actions and self.history[obj_id]:
            accepted_action = self.last_actions[obj_id]
            current_record = self.history[obj_id][-1]
            if current_record["action"] == accepted_action:
                current_record["duration"] = timestamp - current_record["timestamp"]

    def get_arrival_time(self, obj_id):
        return self.arrival_times.get(obj_id)

    def get_departure_time(self, obj_id):
        return self.departure_times.get(obj_id)

    def get_first_seen_time(self, obj_id):
        hist = self.history.get(obj_id)
        if not hist:
            return None
        return hist[0]["timestamp"]

    def get_current_action_with_duration(self, obj_id):
        hist = self.history.get(obj_id)
        if obj_id in self.last_actions and hist:
            current = hist[-1]
            seconds = current["duration"].total_seconds()
            return f"{current['action']} ({seconds:.1f}s)"
        return "Unknown"

    def get_recent_actions_summary(self, obj_id, max_actions=3):
        hist = self.history.get(obj_id)
        if not hist:
            return "No history"

        recent = list(hist)[-max_actions:]
        summary_parts = []
        for record in recent:
            duration_str = f"{record['duration'].total_seconds():.1f}s"
            time_str = record["timestamp"].strftime("%H:%M:%S")
            summary_parts.append(f"{record['action']} ({duration_str}) at {time_str}")
        return " → ".join(summary_parts)


def setup_streamlit_ui():
    st.set_page_config(page_title="DFN – СИБИНТЕК", layout="wide")

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
        st.markdown('<div class="block-yellow">ЛЮДИ</div>', unsafe_allow_html=True)
        people_placeholder = st.empty()

    with train_col:
        st.markdown('<div class="block-dark">ПОЕЗДА</div>', unsafe_allow_html=True)
        trains_placeholder = st.empty()

    return time_placeholder, video_placeholder, people_placeholder, trains_placeholder


def run_dashboard():
    time_placeholder, video_placeholder, people_placeholder, trains_placeholder = setup_streamlit_ui()

    position_history = collections.defaultdict(
        lambda: collections.deque(maxlen=POSITION_HISTORY_LEN)
    )
    size_history = collections.defaultdict(
        lambda: collections.deque(maxlen=SIZE_HISTORY_LEN)
    )

    trackid_to_global = {}
    global_state = {}
    next_person_id = 1
    next_train_id = 1
    main_train_id = None
    frame_idx = 0

    action_history = ActionHistory()

    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        stream=True,
        show=False,
        tracker="bytetrack.yaml",
        classes=[CLASS_PERSON, CLASS_TRAIN],
        persist=True,
        device=DEVICE,
        imgsz=YOLO_IMGSZ,
        conf=0.25,
        iou=0.5,
        vid_stride=YOLO_VID_STRIDE,
        verbose=False,
    )

    for result in results:
        frame_idx += 1
        frame = result.orig_img.copy()
        frame_h = frame.shape[0]

        now = datetime.now()
        time_placeholder.markdown(
            f'<div class="big-time">{now.strftime("%H:%M")}</div>',
            unsafe_allow_html=True,
        )

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
                except Exception:
                    track_id = int(box.id.item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                center = (cx, cy)

                obj_class = int(box.cls.item())
                bbox_height = float(y2 - y1)
                conf_score = float(box.conf.item())

                if obj_class == CLASS_PERSON:
                    local_thr = get_person_conf_threshold(center, frame_idx, global_state)
                    if conf_score < local_thr:
                        continue
                elif obj_class == CLASS_TRAIN:
                    rel_h = bbox_height / frame_h
                    if conf_score < TRAIN_CONF_THRESHOLD or rel_h < MIN_TRAIN_REL_HEIGHT:
                        continue
                else:
                    continue

                if track_id in trackid_to_global:
                    global_id = trackid_to_global[track_id]
                else:
                    match = find_matching_global_id(
                        center, bbox_height, frame_idx, obj_class, global_state
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
                    action = analyze_train_movement(
                        position_history[global_id],
                        size_history[global_id],
                    )
                    color = (0, 255, 255)
                    label = f"Train {global_id}: {action}"

                    action_history.record_action(global_id, action, now)

                    arrival_str = format_time(action_history.get_arrival_time(global_id))
                    departure_str = format_time(action_history.get_departure_time(global_id))

                    trains_out.append(
                        {
                            "ID": global_id,
                            "Status": action,
                            "Arrived": arrival_str,
                            "Departed": departure_str,
                            "Current Action": action_history.get_current_action_with_duration(global_id),
                        }
                    )
                else:
                    movement_action = analyze_person_movement(
                        position_history[global_id], EFFECTIVE_FPS
                    )
                    size_action = "Close" if bbox_height > frame_h * 0.4 else "Far"
                    action = f"{movement_action} ({size_action})"
                    color = (0, 255, 0)
                    label = f"Person {global_id}: {action}"

                    action_history.record_action(global_id, action, now)

                    first_seen_time = action_history.get_first_seen_time(global_id)
                    first_seen_str = format_time(first_seen_time) if first_seen_time else "N/A"

                    people_out.append(
                        {
                            "ID": global_id,
                            "Action": action_history.get_current_action_with_duration(global_id),
                            "History": action_history.get_recent_actions_summary(global_id),
                            "First Seen": first_seen_str,
                            "Frame": frame_idx,
                        }
                    )

                global_state[global_id] = {
                    "last_pos": center,
                    "last_frame": frame_idx,
                    "last_action": action,
                    "class": obj_class,
                    "bbox": (x1, y1, x2, y2),
                }

                if frame_idx % SEND_EVERY_N_FRAMES == 10:
                    db_queue.put(
                        {
                            "person_id": str(global_id),
                            "frame": int(frame_idx),
                            "action": action,
                            "x": float(cx),
                            "y": float(cy),
                        }
                    )

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
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
            max_missed = MISSED_FRAMES.get(obj_class, MISSED_FRAMES[CLASS_PERSON])

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
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
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
                first_seen_str = format_time(first_seen_time) if first_seen_time else "N/A"

                people_out.append(
                    {
                        "ID": gid,
                        "Action": action_history.get_current_action_with_duration(gid),
                        "History": action_history.get_recent_actions_summary(gid),
                        "First Seen": first_seen_str,
                        "Frame": state["last_frame"],
                    }
                )
            elif obj_class == CLASS_TRAIN:
                if gid in displayed_train_ids:
                    continue
                if main_train_id is not None and gid != main_train_id:
                    continue

                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
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

                arrival_str = format_time(action_history.get_arrival_time(gid))
                departure_str = format_time(action_history.get_departure_time(gid))

                trains_out.append(
                    {
                        "ID": gid,
                        "Status": last_action,
                        "Arrived": arrival_str,
                        "Departed": departure_str,
                        "Current Action": action_history.get_current_action_with_duration(gid),
                    }
                )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)

        if people_out:
            df_people = pd.DataFrame(people_out)
            styled_people = df_people.style.set_properties(
                **{"max-width": "300px", "font-size": "12px"},
                subset=["History"],
            )
            people_placeholder.dataframe(
                styled_people, hide_index=True, use_container_width=True
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
                color_train_status, subset=["Status"]
            )
            trains_placeholder.dataframe(
                styled_trains, hide_index=True, use_container_width=True
            )
        else:
            trains_placeholder.write("Нет активных поездов")

    st.subheader("Полная история поездов")
    all_train_ids = [
        obj_id for obj_id in action_history.history.keys() if obj_id.startswith("T")
    ]
    for train_id in all_train_ids:
        arrival = action_history.get_arrival_time(train_id)
        departure = action_history.get_departure_time(train_id)

        st.write(f"**{train_id}**:")
        if arrival:
            st.write(f"  - Прибыл: {format_time(arrival)}")
        if departure:
            st.write(f"  - Уехал: {format_time(departure)}")

    db_queue.join()
    db_queue.put(None)
    db_thread.join()


if __name__ == "__main__":
    run_dashboard()
