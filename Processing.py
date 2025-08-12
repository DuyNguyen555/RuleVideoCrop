import cv2
import config

from Utils import *
from MotionDetect import *
    

def ProcessFrame(frame, kernel, frame_index, motion_detector, recorder):
    if frame_index % 3 != 0:
        return
    
    # Rule resize
    height, width = frame.shape[:2]
    resize_frame = cv2.resize(frame, (width // config.resize_factor, height // config.resize_factor), interpolation=cv2.INTER_LINEAR)
    hsv_resize_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)
    roi = RoiCropped(hsv_resize_frame, resize_frame.shape[0], resize_frame.shape[1])
    # cv2.imshow("ROI", roi)
    
    # Rule Orange Bar
    bar_found, bar_box_points = DetectOrangeBar(frame=roi, kernel=kernel)

    # roi_bgr = RoiCropped(resize_frame, resize_frame.shape[0], resize_frame.shape[1])
    # bar_found, bar_box_points = DetectOrangeBar(frame=roi, kernel=kernel, frame_bgr=roi_bgr)

    # Rule QR Scan
    isQR = False
    if bar_found:
        isQR, qr_results = DetectQRCode(frame, bar_box_points, config.resize_factor)
        if qr_results:
            for r in qr_results:
                if r not in config.ls_qr and r not in config.name_video_saved:
                    config.ls_qr.append(r)
    
    # Rule Motion
    Motion(frame, frame_index, bar_found, bar_box_points, roi, motion_detector, isQR)


    
    # print(f"Frame{frame_index} Current: {config.motion_current}")
    # print(f"Frame{frame_index} Before: {config.motion_before}")

    # print(f"Frame{frame_index} - Start:",config.trigger_start)
    # if config.trigger_stop == True:
    #     print("*"*20)
    # print(f"Frame{frame_index} - End:",config.trigger_stop)
