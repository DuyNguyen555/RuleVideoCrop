import time
import json
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from confluent_kafka import Producer


# Hàm gửi message đến Kafka
def send_to_kafka(file_path):
    conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(conf)

    topic = "videos"
    message = json.dumps({"path": file_path})

    producer.produce(topic, key="file_event", value=message.encode("utf-8"))
    producer.flush() 


# Handler cho watchdog
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            print(f"New file: {event.src_path}")
            # chạy gửi Kafka trên thread riêng
            t = threading.Thread(target=send_to_kafka, args=(event.src_path,))
            t.daemon = True
            t.start()


if __name__ == "__main__":
    path = "./video/output" 
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    print("Loading... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
