import cv2
import time
import json
import multiprocessing
from confluent_kafka import Consumer
from processing import process_video

# def process_video(path):
#     cap = cv2.VideoCapture(path)
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         time.sleep(0.001)
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"[Worker] Done {path}, total frames={frame_count}")

def handle_message(msg):
    raw_value = msg.value()
    print(f"[DEBUG] Raw Kafka message: {raw_value}")

    if raw_value is None:
        print("[Consumer] Nhận được message rỗng, bỏ qua.")
        return

    try:
        data = json.loads(raw_value.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"[Consumer] Message không phải JSON, bỏ qua: {raw_value}")
        return

    path = data.get("path")
    if not path:
        print("[Consumer] Không tìm thấy path trong message, bỏ qua.")
        return

    print(f"[Consumer] Received video path: {path}")

    p = multiprocessing.Process(target=process_video, args=(path,))
    p.start()


if __name__ == "__main__":
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'video-workers',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe(["videos"])

    print("[Consumer] Waiting for messages...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[Consumer] Error: {msg.error()}")
                continue

            handle_message(msg)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
