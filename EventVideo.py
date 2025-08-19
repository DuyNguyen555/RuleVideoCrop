import cv2
import os

class EventVideoRecorder:
    def __init__(self, fps, frame_size, output_dir="output_videos"):
        self.fps = float(fps)
        self.frame_size = frame_size
        self.output_dir = output_dir
        self.is_recording = False
        self.writer = None
        os.makedirs(output_dir, exist_ok=True)

    def start_recording(self, filename):
        if self.is_recording:
            # print("Recording")
            return
        
        print("Start Record")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.writer = cv2.VideoWriter(
            os.path.join(self.output_dir, filename),
            fourcc,
            self.fps,
            self.frame_size
        )
        self.is_recording = True

    def stop_recording(self):
        if self.is_recording and self.writer:
            self.writer.release()

        self.is_recording = False
        print("End Record")

    def update(self, frame):
        if self.is_recording:
            # resize nếu cần
            # if (frame.shape[1], frame.shape[0]) != self.frame_size:
            #     frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)
