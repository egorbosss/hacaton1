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
import os
import tempfile


try:
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    try:
        img = Image.open("number_train/orig.jpg")
        TRAIN_ID_TEXT = pytesseract.image_to_string(
            img,
            lang="rus+eng",
        ).strip()
        if not TRAIN_ID_TEXT:
            TRAIN_ID_TEXT = "Unknown"
    except Exception:
        TRAIN_ID_TEXT = "Unknown"
except ImportError:
    pytesseract = None
    TRAIN_ID_TEXT = "Unknown"

from db import SessionLocal, Detection  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

VIDEO_PATH = "videos/video.mp4"
DET_MODEL_PATH = "yolo11x.pt"
POSE_MODEL_PATH = "yolo11x-pose.pt"

with SessionLocal() as s:
    s.query(Detection).delete()
    s.commit()

FRAME_HEIGHT = 720

YOLO_IMGSZ = 960
YOLO_VID_STRIDE = 3

VIDEO_FPS = 20
EFFECTIVE_FPS = VIDEO_FPS / YOLO_VID_STRIDE

POSITION_HISTORY_LEN = 30
SIZE_HISTORY_LEN = 30

MAX_REID_DISTANCE = 150
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

IDLE_STANDING_SEC = 3.0
IDLE_WALK_SLOW_SEC = 2.0

ACTIVE_KPT_MOVEMENT_REL = 0.015
TORSO_BENT_ANGLE = 20.0

WORK_NEAR_TRAIN_MARGIN_REL = 0.1
TRANSIT_RADIUS_REL = 0.18

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

SKELETON_PAIRS_ARMS = [(5, 7), (7, 9), (6, 8), (8, 10)]
SKELETON_PAIRS_LEGS = [(11, 13), (13, 15), (12, 14), (14, 16)]
SKELETON_PAIRS_BODY = [(5, 11), (6, 12)]

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

                status = item.get("status")
                action_str = item.get("action")
                details = item.get("details")

                merged_parts = []
                if status:
                    merged_parts.append(str(status))
                if action_str:
                    merged_parts.append(str(action_str))
                if details:
                    merged_parts.append(str(details))

                if merged_parts:
                    merged_action = " | ".join(merged_parts)
                else:
                    merged_action = str(item.get("action", ""))

                det = Detection(
                    person_id=pid,
                    frame=item["frame"],
                    action=merged_action,
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


def analyze_person_movement(positions, fps, global_id=None,
                            speed_history=None):
    if len(positions) < 2:
        return "Standing"

    pts = list(positions)
    dist = 0.0

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dist += math.hypot(x1 - x0, y1 - y0)

    time_s = (len(pts) - 1) / fps
    speed = dist / max(time_s, 1e-3)

    if global_id is not None and speed_history is not None:
        hist = speed_history[global_id]
        hist.append(speed)
        speed = sum(hist) / max(len(hist), 1)

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


def get_body_kpt(kpts, confs, idx, thr=0.5):
    if confs is None:
        return None
    if idx >= len(kpts) or idx >= len(confs):
        return None
    if confs[idx] < thr:
        return None

    pt = kpts[idx]
    x, y = float(pt[0]), float(pt[1])

    if x == 0 or y == 0:
        return None
    return x, y


def classify_person_pose(kpts, confs, frame_h):
    if kpts is None or confs is None:
        return "Pose: unknown", {}

    ls = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["left_shoulder"],
    )
    rs = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["right_shoulder"],
    )
    lh = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["left_hip"],
    )
    rh = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["right_hip"],
    )
    lw = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["left_wrist"],
    )
    rw = get_body_kpt(
        kpts,
        confs,
        BODY_KEYPOINT_INDICES["right_wrist"],
    )

    if ls and rs:
        shoulders_mid = (
            (ls[0] + rs[0]) / 2.0,
            (ls[1] + rs[1]) / 2.0,
        )
    else:
        shoulders_mid = None

    if lh and rh:
        hips_mid = (
            (lh[0] + rh[0]) / 2.0,
            (lh[1] + rh[1]) / 2.0,
        )
    else:
        hips_mid = None

    arms_up = False
    if ls and lw:
        if lw[1] < ls[1] - 0.05 * frame_h:
            arms_up = True
    if rs and rw:
        if rw[1] < rs[1] - 0.05 * frame_h:
            arms_up = True

    hands_down = False
    if lh and lw:
        if lw[1] > lh[1] + 0.05 * frame_h:
            hands_down = True
    if rh and rw:
        if rw[1] > rh[1] + 0.05 * frame_h:
            hands_down = True

    torso_angle = None
    is_torso_bent = False

    if shoulders_mid and hips_mid:
        dx = shoulders_mid[0] - hips_mid[0]
        dy = shoulders_mid[1] - hips_mid[1]

        if dy == 0:
            torso_angle = 90.0
        else:
            torso_angle = abs(math.degrees(math.atan2(dx, dy)))

        if torso_angle > TORSO_BENT_ANGLE:
            is_torso_bent = True

        if arms_up:
            pose_label = "Pose: arms up"
        elif torso_angle < 15:
            pose_label = "Pose: standing straight"
        elif torso_angle < 40:
            pose_label = "Pose: leaning"
        else:
            pose_label = "Pose: bending"
    else:
        if arms_up:
            pose_label = "Pose: arms up"
        else:
            pose_label = "Pose: unknown"

    flags = {
        "arms_up": arms_up,
        "hands_down": hands_down,
        "is_torso_bent": is_torso_bent,
        "torso_angle": torso_angle,
    }

    return pose_label, flags


def draw_person_skeleton(frame, kpts, confs):
    if kpts is None or confs is None:
        return

    def valid(idx):
        if idx >= len(kpts) or idx >= len(confs):
            return False
        if confs[idx] <= 0.5:
            return False
        pt = kpts[idx]
        x, y = float(pt[0]), float(pt[1])
        return not (x == 0 or y == 0)

    for start, end in (
        SKELETON_PAIRS_ARMS + SKELETON_PAIRS_LEGS + SKELETON_PAIRS_BODY
    ):
        if not valid(start) or not valid(end):
            continue

        pt1 = kpts[start]
        pt2 = kpts[end]
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def update_work_state(global_id, movement_action, pose_label, frame_idx,
                      person_idle_state, frame_h, pose_flags=None,
                      kpt_movement=0.0):
    pose_flags = pose_flags or {}
    pose_label_lower = pose_label.lower()

    active_by_pose_flags = (
        pose_flags.get("arms_up")
        or pose_flags.get("is_torso_bent")
        or False
    )

    if pose_flags.get("hands_down") and pose_flags.get("is_torso_bent"):
        active_by_pose_flags = True

    active_by_text = any(
        token in pose_label_lower for token in ["arms up", "bending", "leaning"]
    )

    active_by_joints = (
        kpt_movement > ACTIVE_KPT_MOVEMENT_REL * frame_h
    )

    if active_by_pose_flags or active_by_text or active_by_joints:
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
    seconds_idle = frames_idle * YOLO_VID_STRIDE / VIDEO_FPS

    if movement_action == "Standing":
        if seconds_idle >= IDLE_STANDING_SEC:
            return "Not working", seconds_idle
    else:
        if seconds_idle >= IDLE_WALK_SLOW_SEC:
            return "Not working", seconds_idle

    return "Working", seconds_idle


def find_matching_global_id(center, bbox_height, frame_idx, obj_class,
                            global_state):
    best_gid = None
    best_dist = None
    best_gap = None
    cx, cy = center
    max_gap = FRAME_GAP.get(obj_class, FRAME_GAP[CLASS_PERSON])
    k_reid = K_REID.get(obj_class, K_REID[CLASS_PERSON])

    adaptive_dist = bbox_height * 0.45 * k_reid
    max_reid_dist = min(MAX_REID_DISTANCE, adaptive_dist)

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
                abs(dist - best_dist)
                <= 0.1 * max(best_dist, 1e-6)
                and frame_gap < best_gap
            ):
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
    def __init__(self, max_history_per_id=50,
                 stable_frames=ACTION_STABLE_FRAMES):
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
        if not obj_id.startswith("T"):
            return

        if action.startswith("Arrived") and obj_id not in self.arrival_times:
            self.arrival_times[obj_id] = timestamp
        elif (
            action.startswith("Departed")
            and obj_id not in self.departure_times
        ):
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
            summary_parts.append(
                f"{record['action']} ({duration_str}) at {time_str}",
            )

        return " → ".join(summary_parts)


def calculate_kpi_for_person(action_history: ActionHistory,
                             person_id: str) -> float:
    hist = action_history.history.get(person_id)
    if not hist:
        return 0.0

    total_duration = sum((r["duration"] for r in hist), timedelta(0))
    if total_duration.total_seconds() <= 0:
        return 0.0

    working_duration = sum(
        (r["duration"] for r in hist if "Working" in r["action"]),
        timedelta(0),
    )
    kpi = working_duration.total_seconds() / total_duration.total_seconds()
    return round(kpi * 100, 1)


def calculate_global_kpi(action_history: ActionHistory) -> float:
    total_duration = timedelta(0)
    working_duration = timedelta(0)

    for obj_id, hist in action_history.history.items():
        if not str(obj_id).startswith("P"):
            continue
        for record in hist:
            dur = record["duration"]
            if not isinstance(dur, timedelta):
                continue
            total_duration += dur
            if "Working" in record["action"]:
                working_duration += dur

    if total_duration.total_seconds() <= 0:
        return 0.0

    kpi = working_duration.total_seconds() / total_duration.total_seconds()
    return round(kpi * 100, 1)


def setup_streamlit_ui():
    st.set_page_config(page_title="DFN - СИБИНТЕК", layout="wide")
    st.markdown(
        """
    <style>
    .stMainBlockContainer{
        background-color: #FFFFFF;
    }
    p {
        color: #201600;
    }
    .stMain {
        background-color: #FFFFFF;
    }
    .block-container {
        padding-top: 0.5rem;
        margin-top: -2rem;
    }
    .block-yellow {
        background-color: #f6ce45;
        padding: 10px;
        color: #000;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 0px;
    }
    .block-dark {
        background-color: #202020;
        padding: 10px;
        color: #fff;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        border-radius: 12px;
        margin-bottom: 0px;
    }
    .big-time {
        font-size: 52px;
        font-weight: 900;
        text-align: center;
        color: #0e0f10;
    }
    .stAppToolbar, .stAppHeader{
        visibility: hidden;
    }
    div.stButton > button {
        background-color: #d40000 !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 2px solid #8b0000 !important;
        padding: 8px 16px !important;
    }
    div.stButton > button:hover {
        background-color: #ff1a1a !important;
        color: white !important;
        border: 2px solid #a00000 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if "stop_dashboard" not in st.session_state:
        st.session_state["stop_dashboard"] = False

    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

    top_left, top_center, top_right = st.columns([2, 3, 1])

    with top_center:
        time_placeholder = st.empty()
    with top_left:
        st.image("static/logo1.svg", width=350)
    with top_right:
        if st.button("Остановить программу", key="stop_button"):
            st.session_state["stop_dashboard"] = True

    row2_left, row2_right = st.columns([2, 2])

    with row2_left:
        video_placeholder = st.empty()
    with row2_right:
        st.markdown(
            '<div class="block-yellow">глобальный KPI</div>',
            unsafe_allow_html=True,
        )
        kpi_chart_placeholder = st.empty()

    row3_left, row3_right = st.columns([1, 1])

    with row3_left:
        st.markdown(
            '<div class="block-yellow">люди</div>',
            unsafe_allow_html=True,
        )
        people_placeholder = st.empty()

    with row3_right:
        col_trains, col_log = st.columns([1, 1])

    with col_trains:
        st.markdown(
            '<div class="block-dark">поезда</div>',
            unsafe_allow_html=True,
        )
        trains_placeholder = st.empty()

    with col_log:
        st.markdown(
            '<div class="block-dark">log</div>',
            unsafe_allow_html=True,
        )
        log_placeholder = st.empty()

    return (
        time_placeholder,
        video_placeholder,
        kpi_chart_placeholder,
        people_placeholder,
        trains_placeholder,
        log_placeholder,
    )


def extract_video_start_time(frame):
    return None


def point_rect_distance(x, y, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    dx = 0.0
    if x < x1:
        dx = x1 - x
    elif x > x2:
        dx = x - x2

    dy = 0.0
    if y < y1:
        dy = y1 - y
    elif y > y2:
        dy = y - y2

    return math.hypot(dx, dy)


def run_dashboard():
    (
        time_placeholder,
        video_placeholder,
        kpi_chart_placeholder,
        people_placeholder,
        trains_placeholder,
        log_placeholder,
    ) = setup_streamlit_ui()

    if st.session_state.get("stop_dashboard"):
        st.warning(
            "Мониторинг остановлен. "
            "Перезапустите приложение для нового запуска.",
        )
        return

    position_history = collections.defaultdict(
        lambda: collections.deque(maxlen=POSITION_HISTORY_LEN),
    )
    size_history = collections.defaultdict(
        lambda: collections.deque(maxlen=SIZE_HISTORY_LEN),
    )

    joint_history = collections.defaultdict(
        lambda: collections.deque(maxlen=10),
    )
    speed_history = collections.defaultdict(
        lambda: collections.deque(maxlen=5),
    )

    trackid_to_global = {}
    global_state = {}
    next_person_id = 1
    next_train_id = 1
    main_train_id = None
    frame_idx = 0
    action_history = ActionHistory()
    person_idle_state = {}

    kpi_history = []
    log_rows = []

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()

    if ret:
        video_start_time = extract_video_start_time(first_frame)
    else:
        video_start_time = None

    if video_start_time is None:
        video_start_time = datetime.now()

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

        elapsed_seconds = (frame_idx - 1) * YOLO_VID_STRIDE / VIDEO_FPS
        frame_time = video_start_time + timedelta(seconds=elapsed_seconds)

        time_placeholder.markdown(
            f'<div class="big-time">{frame_time.strftime("%H:%M")}</div>',
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

                db_status = None
                db_action_str = None
                db_details = None

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

                    current_action_train = (
                        action_history
                        .get_current_action_with_duration(global_id)
                    )

                    trains_out.append(
                        {
                            "ID": TRAIN_ID_TEXT,
                            "Status": train_action,
                            "Arrived": arrival_str,
                            "Departed": departure_str,
                            "Current Action": current_action_train,
                        },
                    )

                    db_status = train_action
                    db_action_str = current_action_train
                    db_details = f"Train {global_id}: {train_action}"
                else:
                    crop = frame[y1:y2, x1:x2]
                    pose_label = "Pose: unknown"
                    pose_flags = {}
                    kpt_movement = 0.0

                    if crop.size > 0:
                        pose_res_list = pose_model(
                            crop,
                            imgsz=YOLO_IMGSZ,
                            verbose=False,
                        )
                        if len(pose_res_list) > 0:
                            pose_res = pose_res_list[0]
                            if (
                                pose_res.keypoints is not None
                                and len(pose_res.keypoints) > 0
                            ):
                                kpts = (
                                    pose_res.keypoints.data[0]
                                    .cpu()
                                    .numpy()
                                )
                                if (
                                    hasattr(
                                        pose_res.keypoints,
                                        "conf",
                                    )
                                    and pose_res.keypoints.conf is not None
                                ):
                                    confs = (
                                        pose_res.keypoints.conf[0]
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    confs = np.ones(
                                        kpts.shape[0],
                                        dtype=np.float32,
                                    )

                                kpts_vis = kpts.copy()
                                if kpts_vis.shape[1] >= 2:
                                    kpts_vis[:, 0] += x1
                                    kpts_vis[:, 1] += y1

                                if kpts_vis.shape[0] > 0:
                                    ys = kpts_vis[:, 1].astype(np.float32)
                                    hist = joint_history[global_id]
                                    hist.append(ys)
                                    if len(hist) >= 2:
                                        kpt_movement = float(
                                            np.mean(
                                                np.abs(hist[-1] - hist[-2]),
                                            ),
                                        )

                                pose_label, pose_flags = classify_person_pose(
                                    kpts,
                                    confs,
                                    frame_h,
                                )

                    movement_action = analyze_person_movement(
                        position_history[global_id],
                        EFFECTIVE_FPS,
                        global_id,
                        speed_history,
                    )
                    size_action = (
                        "Close"
                        if bbox_height > frame_h * 0.4
                        else "Far"
                    )

                    in_train_box = False
                    near_train = False
                    transit_mode = False

                    if main_train_id is not None and main_train_id in global_state:
                        tx1, ty1, tx2, ty2 = global_state[main_train_id]["bbox"]

                        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                            in_train_box = True

                        dist_to_train = point_rect_distance(
                            cx,
                            cy,
                            (tx1, ty1, tx2, ty2),
                        )
                        margin = WORK_NEAR_TRAIN_MARGIN_REL * frame_h
                        if dist_to_train <= margin:
                            near_train = True

                    traj = position_history[global_id]
                    if len(traj) >= 10:
                        xs = [p[0] for p in traj]
                        ys = [p[1] for p in traj]
                        mx = sum(xs) / len(xs)
                        my = sum(ys) / len(ys)
                        max_dist = max(
                            math.hypot(x - mx, y - my) for x, y in traj
                        )
                        diag = math.hypot(frame_w, frame_h)
                        radius_thr = TRANSIT_RADIUS_REL * diag
                        if max_dist > radius_thr:
                            transit_mode = True

                    work_state, idle_seconds = update_work_state(
                        global_id,
                        movement_action,
                        pose_label,
                        frame_idx,
                        person_idle_state,
                        frame_h,
                        pose_flags=pose_flags,
                        kpt_movement=kpt_movement,
                    )

                    if in_train_box:
                        if global_id in person_idle_state:
                            person_idle_state.pop(global_id, None)
                        work_state = "Working"
                        idle_seconds = 0.0
                    else:
                        if (
                            movement_action in ("Walking", "Moving fast")
                            and transit_mode
                            and not near_train
                        ):
                            if global_id in person_idle_state:
                                person_idle_state.pop(global_id, None)
                            work_state = "Not working"

                    detail = (
                        f"{movement_action} ({size_action}) | "
                        f"{pose_label} | idle={idle_seconds:.1f}s | "
                        f"in_train_box={in_train_box} | "
                        f"near_train={near_train} | transit={transit_mode}"
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
                    first_seen_str = (
                        format_time(first_seen_time)
                        if first_seen_time
                        else "N/A"
                    )

                    kpi_value = calculate_kpi_for_person(
                        action_history,
                        global_id,
                    )

                    current_action_person = (
                        action_history
                        .get_current_action_with_duration(global_id)
                    )

                    people_out.append(
                        {
                            "ID": global_id,
                            "Work": work_state,
                            "KPI": kpi_value,
                            "Action": current_action_person,
                            "Details": detail,
                            "First Seen": first_seen_str,
                            "Frame": frame_idx,
                        },
                    )

                    db_status = work_state
                    db_action_str = current_action_person
                    db_details = detail

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
                            "status": db_status,
                            "action": db_action_str,
                            "details": db_details,
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
                first_seen_str = (
                    format_time(first_seen_time)
                    if first_seen_time
                    else "N/A"
                )
                kpi_value = calculate_kpi_for_person(
                    action_history,
                    gid,
                )

                people_out.append(
                    {
                        "ID": gid,
                        "Work": last_action,
                        "KPI": kpi_value,
                        "Action": (
                            action_history
                            .get_current_action_with_duration(gid)
                        ),
                        "Details": "",
                        "First Seen": first_seen_str,
                        "Frame": state["last_frame"],
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
                        "ID": TRAIN_ID_TEXT,
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

        global_kpi = calculate_global_kpi(action_history)
        elapsed_total_sec = (frame_time - video_start_time).total_seconds()
        bucket_10s = int(elapsed_total_sec // 10)

        kpi_history.append(
            {
                "bucket_10s": bucket_10s,
                "kpi": global_kpi,
            },
        )
        df_kpi = pd.DataFrame(kpi_history)
        df_kpi = (
            df_kpi.groupby("bucket_10s", as_index=True)["kpi"]
            .mean()
            .sort_index()
        )
        df_kpi.index = df_kpi.index * 10
        df_kpi.index.name = "seconds"

        with kpi_chart_placeholder.container():
            st.markdown(
                """
                <div style="
                    padding: 15px;
                    border-radius: 15px;
                ">
                """,
                unsafe_allow_html=True,
            )
            st.line_chart(df_kpi)
            st.markdown("</div>", unsafe_allow_html=True)

        if frame_idx % 10 == 0:
            log_rows.append(
                {
                    "time": frame_time.strftime("%H:%M:%S"),
                    "frame": frame_idx,
                    "people": len(people_out),
                    "trains": len(trains_out),
                    "kpi": global_kpi,
                },
            )
            df_log = pd.DataFrame(log_rows[-50:])
            log_placeholder.dataframe(
                df_log,
                hide_index=True,
                use_container_width=True,
            )

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

    db_queue.join()
    db_queue.put(None)
    db_thread.join()


if __name__ == "__main__":
    run_dashboard()
