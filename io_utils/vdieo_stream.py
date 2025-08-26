import cv2
import time
from threading import Thread
from queue import Queue
import cv2
import time
from threading import Thread
from queue import Queue

class VideoStream:
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
             raise RuntimeError(f"Cannot open video: {path}")
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)
        
        # Tạo và cấu hình luồng
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True # Luồng sẽ tự động thoát khi chương trình chính kết thúc

    def start(self):
        """Bắt đầu luồng đọc frame."""
        self.thread.start()
        return self

    def update(self):
        """
        Vòng lặp chạy trong luồng riêng.
        Chỉ đọc frame và đưa vào Queue.
        """
        while not self.stopped:
            # Nếu queue chưa đầy, đọc frame mới
            if not self.Q.full():
                ret, frame = self.stream.read()
                
                # Nếu không đọc được (hết video), set cờ dừng và thoát vòng lặp
                if not ret:
                    self.stopped = True
                    break
                
                self.Q.put(frame)
            else:
                # Nếu queue đầy, nghỉ một chút để tránh chiếm 100% CPU
                time.sleep(0.01)
        
        # Khi vòng lặp kết thúc, luồng sẽ tự động dừng lại.
        self.stream.release()

    def read(self):
        """Lấy một frame từ hàng đợi. Hàm này sẽ block cho đến khi có frame."""
        return self.Q.get()

    def running(self):
        """Kiểm tra xem luồng có còn đang chạy hay không."""
        # Luồng được coi là đang chạy nếu nó chưa bị set cờ stop HOẶC queue vẫn còn frame
        return not self.stopped or not self.Q.empty()

    def stop(self):
        """
        Set cờ để báo cho luồng dừng lại.
        Hàm này được gọi từ luồng chính.
        """
        self.stopped = True
