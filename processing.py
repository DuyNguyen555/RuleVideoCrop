import os
import time
import cv2
import csv

import config
from state import State
from pipeline import Pipeline
from vision.motion import MotionDetector
from actions.snapshot import Snapshot
from actions.renamer import Rename
from vision.qr import QRDetector

def process_video(src):
    print("Run: ", src)

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output"
    name_video =  os.path.splitext(os.path.basename(src))[0]
    output_img = os.path.join(output_path, name_video, "img")
    os.makedirs(output_img, exist_ok=True)

    state = State()
    state.OUTPUT_IMG = output_img

    pipe = Pipeline(state=state, motion_detector=MotionDetector(), qr_detector=QRDetector(), snapshot=Snapshot(),rename=Rename())

    # kernel size
    k = max(1, round(30 / config.RESIZE_FACTOR) - 2)
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    frame_index = 0
    # chu kỳ log
    N = 100

    log_path = f"{output_path}/{name_video}/log_time.csv"
    # log_path = getattr(config, "LOG_TIME_CSV", "log_time.csv")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fcsv = open(log_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["frame", "read_ms", "orange_ms", "white_ms", "qr_ms", "motion_ms", "save_img_ms", "rename_file_ms", "proc_ms", "total_ms"])

    while cap.isOpened():
        t0 = time.perf_counter()
        if frame_index % 2 != 0:
            cap.grab()
            frame_index += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break
        read_time = time.perf_counter() - t0
        # print(read_time)
        t1 = time.perf_counter()
        o_t, w_t, qr_t, mt_t, si_t, re_t  = pipe.process_frame(frame, kernel, frame_index)
        proc_time = time.perf_counter() - t1

        total = read_time + proc_time

        writer.writerow([
            frame_index,
            f"{read_time*1000:.4f}",
            f"{o_t*1000:.4f}",
            f"{w_t*1000:.4f}",
            f"{qr_t*1000:.4f}", 
            f"{mt_t*1000:.4f}", 
            f"{si_t*1000:.4f}", 
            f"{re_t*1000:.4f}", 
            f"{proc_time*1000:.4f}", 
            f"{total*1000:.4f}",
        ])

        # flush theo chu kỳ
        if frame_index % N == 0:
            fcsv.flush()

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_index += 1



    print("="*40)
    print(f"Video: {src}")
    print(f"Resolution: {width}x{height} | FPS(meta): {fps:.2f}")
    # print(f"Total elapsed: {elapsed_total_s:.3f} s")
    print(f"CSV saved to: {os.path.abspath(log_path)}")
    print("="*40)

    cap.release()
    fcsv.close()
    cv2.destroyAllWindows()
