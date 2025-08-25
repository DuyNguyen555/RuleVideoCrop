import cv2
import math
import numpy as np
from typing import Tuple
import config
from state import State

from vision.color_bar import detect_orange_bar, detect_white
from vision.qr import detect_qr
from vision.motion import MotionDetector, FindMotion
from vision.draw import draw_thresholds
from actions.snapshot import save_snapshot, snapshot_video
from vision.geometry import roi_bounds
from actions.renamer import rename_pair_from_queue

class Pipeline:
    def __init__(self, state: State, motion_detector: MotionDetector):
        self.state = state
        self.motion_detector = motion_detector

    def _sort_qr_motion(self, motion):
        if motion == "Down":
            return True
        elif motion == "Up":
            return False
        
    def _qr_base(self, qr: str) -> str:
        return qr[:-1]

    def _has_tail(self, base: str, tail: str) -> bool:
        return any(s.startswith(base) and s.endswith(tail) for s in self.state.ls_qr)
    
    def _rename_if_ready(self):
        # if self.state.wait_pair_active:
        #     return
        if self.state.motion_current not in ("Down", "Up"):
            return
        
        # nếu đã đủ 2 ảnh cho pallet đầu và có QR
        imgs, ls_qr, saved, new_pallet_seq, changed = rename_pair_from_queue(
            config.OUTPUT_IMG, self.state.imgs_save, self.state.ls_qr, self.state.name_video_saved, self.state.pallet_seq, self.state.motion_current
        )
        self.state.imgs_save = imgs
        self.state.ls_qr = ls_qr
        self.state.name_video_saved = saved
        self.state.pallet_seq = new_pallet_seq
        # if changed:
        #     self.state.is_rename_img = False

    def process_frame(self, frame, kernel, frame_index: int):
        if frame_index % 2 == 0:
            return 0, 0, 0, 0, 0
        
        # Rule ROI
        if not self.state.roi_ready:
            self.state.roi_x1, self.state.roi_y1, self.state.roi_x2, self.state.roi_y2 = roi_bounds(frame_shape=frame.shape)
            self.state.roi_ready = True
        
        roi = frame[self.state.roi_y1:self.state.roi_y2, self.state.roi_x1:self.state.roi_x2]

        resized = cv2.resize(roi, (roi.shape[1] // config.RESIZE_FACTOR,
                                   roi.shape[0] // config.RESIZE_FACTOR),
                                    interpolation=cv2.INTER_LINEAR)
        hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # self._init_thresholds(resized.shape[0])

        # Rule Orange bar
        frame_bar, y_bar, orange_time = detect_orange_bar(hsv=hsv_resized,
                                                      kernel=kernel,
                                                      show_result=False)
                
        # Rule White in Orange bar
        white_frame, boxes, white_time = detect_white(frame_bar, roi, show_result=True)

        # Rule QRcode
        qr_time = detect_qr(white_frame, self.state.ls_qr, self.state.name_video_saved)

        # Rule Motion
        self.state.motion_current, self.state.departure, motion_time = FindMotion(motion_detector=self.motion_detector, 
                                                                            roi=roi,
                                                                            y_bar=y_bar,
                                                                            state=self.state)
        # self.state.motion_current, motion_time = FindMotion(self.motion_detector, roi)
        # print(self.state.motion_current)

        # Rule Snapshot
        save_img_time = snapshot_video(frame=frame,
                        index_frame=frame_index,
                        resized_h=resized.shape[0],
                        y_bar=y_bar,
                        state=self.state)
        

        # Rule Rename
        self._rename_if_ready()

        return orange_time, white_time, qr_time, motion_time, save_img_time
    


