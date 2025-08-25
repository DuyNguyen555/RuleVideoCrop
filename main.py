import os
import time
import cv2
import numpy as np
import csv


import config
from state import State
from pipeline import Pipeline
from vision.motion import MotionDetector

def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {config.VIDEO_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state = State(
        frame_stop_scan=config.FRAME_STOP_SCAN,
        time_reset_crop_img=config.TIME_RESET_CROP_IMG,
        rearm_cooldown_frames=config.REARM_COOLDOWN_FRAMES,
        time_rename_img=config.TIME_RENAME_IMG_DEFAULT
    )

    pipe = Pipeline(state=state, motion_detector=MotionDetector())

    # kernel size
    k = max(1, round(30 / config.RESIZE_FACTOR) - 2)
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    frame_index = 0
    # chu kỳ log
    N = 31

    log_path = getattr(config, "LOG_TIME_CSV", "log_time.csv")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fcsv = open(log_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow(["frame", "read_ms", "orange_ms", "white_ms", "qr_ms", "motion_ms", "save_img_ms", "proc_ms", "total_ms"])

    t_total_start = time.perf_counter()
    total_frames = 0

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        read_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        o_t, w_t, qr_t, mt_t, si_t = pipe.process_frame(frame, kernel, frame_index)
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
            f"{proc_time*1000:.4f}", 
            f"{total*1000:.4f}",
        ])

        # flush theo chu kỳ để an toàn dữ liệu nếu dừng giữa chừng
        if frame_index % (N*10) == 0:
            fcsv.flush()

        # if frame_index % N == 0:
        #     print(f"[Frame {frame_index}] read={read_time*1000:.2f} ms | proc={proc_time*1000:.2f} ms | total={total*1000:.2f} ms")
            # print("Ls qr:", state.ls_qr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    # t_total_end = time.perf_counter()
    # elapsed_total_s = t_total_end - t_total_start

    print("="*40)
    print(f"Video: {config.VIDEO_PATH}")
    print(f"Resolution: {width}x{height} | FPS(meta): {fps:.2f}")
    # print(f"Total elapsed: {elapsed_total_s:.3f} s")
    print(f"CSV saved to: {os.path.abspath(log_path)}")
    print("="*40)

    cap.release()
    fcsv.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
