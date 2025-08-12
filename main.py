import cv2
import os
import time
import psutil

import config
from Utils import *
from Processing import ProcessFrame
from MotionDetect import MotionDetector
from EventVideo import EventVideoRecorder

if __name__ == "__main__":
    cap = cv2.VideoCapture(config.video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Goi cac lop phu tro
    recorder = EventVideoRecorder(fps, (width, height), output_dir=config.output_dir)
    motion_detector = MotionDetector()

    frame_index = 0
    frame_clear_video_saved = 0

    kernel_size = max(1, round(30 / config.resize_factor - 2))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    start_frame_time = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        ProcessFrame(frame, kernel, frame_index, motion_detector, recorder)

        len_qr = len(config.ls_qr)
        if len_qr > 1:
            config.ls_qr.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)

        # Kiểm tra trigger
        if getattr(config, "trigger_start", False):
            name_video = f"{config.output_dir}/{config.name_record}.mp4"
            recorder.start_recording(f"{config.name_record}.mp4")
            config.trigger_start = False
                
            temp_video_path = os.path.join(name_video)
            if config.name_record_start == "":
                config.name_record_start = temp_video_path


        if getattr(config, "trigger_stop", False):
            recorder.stop_recording()
            config.trigger_stop = False

            if config.name_record_start != "":
                config.name_record_save.append(config.name_record_start)
                config.name_record_start = ""
                config.len_record += 1

        # print("Length QR",len(config.ls_qr))
        # print("Length Video",len(config.name_record_save))

        # Change name video to QRCode
        RenameVideo(len_qr)

        if config.motion_current in ["Left", "Right"]:
            frame_clear_video_saved += 1
            if frame_clear_video_saved > config.threshold_clear_name_save:
                config.name_video_saved.clear()
                frame_clear_video_saved = 0
       
        # print(config.name_record_save)
        # print(f"Before: {config.motion_before}")
        # print(f"Current: {config.motion_current}")
        # print(config.ls_qr)
        # print(config.name_video_saved)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("QR:", config.ls_qr)
            recorder.stop_recording()
            break
        
        recorder.update(frame)
        frame_index += 1

    cap.release()
    recorder.stop_recording()

    elapsed = (time.perf_counter() - start_frame_time)
    print(f"Total Time: {elapsed:.3f} s")
    cv2.destroyAllWindows()

