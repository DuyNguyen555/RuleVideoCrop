import cv2
import time
import json
import multiprocessing
from confluent_kafka import Consumer
from processing import process_video

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

    # Lấy danh sách video
    video_list = data.get("videos")
    if not video_list or not isinstance(video_list, list):
        print("[Consumer] Không tìm thấy danh sách video trong message, bỏ qua.")
        return

    print(f"[Consumer] Received {len(video_list)} video(s)")

    for path in video_list:
        if not path or not path.lower().endswith(".mp4"):
            print(f"[Consumer] Bỏ qua đường dẫn không hợp lệ: {path}")
            continue

        print(f"[Consumer] Xử lý video: {path}")
        # Tạo tiến trình riêng cho từng video
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
