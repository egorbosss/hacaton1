from ultralytics import YOLO
import cv2
import collections

model = YOLO('yolo11n.pt')
results = model.track(
    source='Drones.mp4',
    show=False,
    tracker="bytetrack.yaml",
    classes=[0],
    persist=True,stream=True
)

# Для хранения истории позиций
position_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
tracked_actions = {}


def analyze_movement(positions):
    """Анализируем движение по истории позиций"""
    if len(positions) < 2:
        return "Stationary"

    # Вычисляем смещение
    first_pos = positions[0]
    last_pos = positions[-1]

    dx = last_pos[0] - first_pos[0]
    dy = last_pos[1] - first_pos[1]
    total_movement = (dx ** 2 + dy ** 2) ** 0.5

    if total_movement < 5:
        return "Standing"
    elif total_movement < 20:
        return "Walking slowly"
    elif total_movement < 50:
        return "Walking"
    else:
        return "Moving fast"


for result in results:
    frame = result.orig_img.copy()

    if result.boxes is not None:
        for box in result.boxes:
            if box.id is not None:
                obj_id = int(box.id)
                bbox = box.xyxy[0].cpu().numpy()

                # Вычисляем центр bounding box
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Сохраняем позицию
                position_history[obj_id].append((center_x, center_y))

                # Анализируем движение
                movement_action = analyze_movement(position_history[obj_id])

                # Определяем общее действие
                bbox_height = y2 - y1
                if bbox_height > frame.shape[0] * 0.4:
                    size_action = "Close"
                else:
                    size_action = "Far"

                action = f"{movement_action} ({size_action})"
                tracked_actions[obj_id] = action

                # Рисуем на кадре
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                info_text = f"ID {obj_id}: {action}"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Общая информация
    cv2.putText(frame, f"People: {len(tracked_actions)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    cv2.imshow('Advanced Action Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\n=== РЕЗУЛЬТАТЫ ТРЕКИНГА ===")
for obj_id, action in tracked_actions.items():
    print(f"Человек {obj_id}: {action}")