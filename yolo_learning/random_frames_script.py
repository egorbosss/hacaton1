import cv2
import os

VIDEOS_DIR = "learning_videos"
OUTPUT_DIR = "cvat_frames"
FRAME_STEP = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(VIDEOS_DIR):
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    video_path = os.path.join(VIDEOS_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        continue

    print(f"Обработка: {video_path}")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            name = f"{os.path.splitext(filename)[0]}_{frame_idx:06d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, name)
            cv2.imwrite(out_path, frame)

        frame_idx += 1

    cap.release()

print("Готово")
