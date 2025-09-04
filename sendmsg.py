from confluent_kafka import Producer
import json

conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

videos = {
    "videos": [
        r"input\part_0120.mp4",
        r"input\part_0030.mp4"
    ]
}

producer.produce("videos", key="file_event", value=json.dumps(videos).encode("utf-8"))
producer.flush()
