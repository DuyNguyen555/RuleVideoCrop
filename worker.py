import cv2
import time
import json
import multiprocessing
from confluent_kafka import Consumer
from processing import process_video
from io_utils.print_log import log


def handle_message(msg):
    raw_value = msg.value()
    if not raw_value:
        log("CONSUMER", "Nhận được message rỗng, bỏ qua.")
        return

    try:
        data = json.loads(raw_value.decode("utf-8"))
    except json.JSONDecodeError:
        log("CONSUMER", f"Message is not JSON, skipping: {raw_value}")
        return

    video_list = data.get("videos")
    if not isinstance(video_list, list):
        log("CONSUMER", "No valid video listing found, skipping.")
        return
    
    # Lọc ra các đường dẫn video hợp lệ
    valid_videos = [
        path for path in video_list
        if isinstance(path, str) and path.lower().endswith(".mp4")
    ]

    if not valid_videos:
        log("CONSUMER", "There are no valid videos in the message.")
        return
        
    log("CONSUMER", f"Nhận được {len(valid_videos)} video(s) để xử lý.")

    # Giới hạn số tiến trình
    num_workers = max(1, multiprocessing.cpu_count() - 1)    
    log("CONSUMER", f"Bắt đầu xử lý với tối đa {num_workers} tiến trình đồng thời...")
    t0 = time.perf_counter()

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_video, valid_videos)

    t1 = time.perf_counter()
    log("CONSUMER", f"Hoàn thành xử lý {len(valid_videos)} video trong {t1 - t0:.2f} giây.")
    print("=" * 60)


if __name__ == "__main__":
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'video-workers',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe(["videos"])

    log("CONSUMER", "Waiting for message from Kafka...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log("ERROR", msg.error())
                continue
            
            handle_message(msg)
    except KeyboardInterrupt:
        log("CONSUMER", "Dừng bởi người dùng.")
    finally:
        consumer.close()