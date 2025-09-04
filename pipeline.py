import cv2
import time
import config
from state import State

from vision.color_bar import detect_orange_bar, detect_white
from vision.qr import QRDetector
from vision.motion import MotionDetector, FindMotion
from actions.snapshot import Snapshot
from vision.geometry import roi_bounds
from actions.renamer import Rename

class Pipeline:
    def __init__(self, state: State, motion_detector: MotionDetector, qr_detector: QRDetector, snapshot: Snapshot, rename: Rename):
        self.state = state
        self.motion_detector = motion_detector
        self.qr_detector = qr_detector
        self.snapshot = snapshot
        self.rename = rename
    
    def _rename_if_ready(self):
        t0 = time.perf_counter()
        if self.state.motion_current not in ("Down", "Up"):
            return time.perf_counter() - t0
        
        # nếu đã đủ 2 ảnh cho pallet đầu và có QR
        imgs, ls_qr, saved, new_pallet_seq, changed = self.rename.rename_pair_from_queue(
            output_img_dir=self.state.OUTPUT_IMG, 
            imgs_save=self.state.imgs_save, 
            ls_qr=self.state.ls_qr, 
            name_video_saved=self.state.name_video_saved, 
            pallet_seq=self.state.pallet_seq, 
            motion=self.state.motion_current
        )
        self.state.imgs_save = imgs
        self.state.ls_qr = ls_qr
        self.state.name_video_saved = saved
        self.state.pallet_seq = new_pallet_seq

        return time.perf_counter() - t0
        # if changed:
        #     self.state.is_rename_img = False

    def process_frame(self, frame, kernel, frame_index: int):
        # if frame_index % 2 == 0:
        #     return 0, 0, 0, 0, 0, 0
            
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
        white_frame, boxes, white_time = detect_white(frame_bar, roi, show_result=False)

        # Rule QRcode
        if self.state.has_found_qr:
            self.state.frame_stop_scan_qr += 1
            if self.state.frame_stop_scan_qr >= 20:
                self.state.has_found_qr = False

        self.state.has_found_qr, qr_time = self.qr_detector.detect_qr(frames=white_frame, 
                                                        state=self.state)
        # self.state.has_found_qr, qr_time = self.qr_detector.detect_qr(frames=white_frame, 
        #                                                 ls_qr=self.state.ls_qr, 
        #                                                 name_video_saved=self.state.name_video_saved,
        #                                                 motion=self.state.motion_current)

        # Rule Motion
        self.state.motion_current, self.state.departure, motion_time = FindMotion(motion_detector=self.motion_detector, 
                                                                                roi=resized,
                                                                                y_bar=y_bar,
                                                                                state=self.state)
        # print(self.state.motion_current)

        # Rule Snapshot
        save_img_time = self.snapshot.snapshot_video(frame=frame,
                        index_frame=frame_index,
                        resized_h=resized.shape[0],
                        y_bar=y_bar,
                        state=self.state)
        

        # Rule Rename
        rename_time = self._rename_if_ready()

        return orange_time, white_time, qr_time, motion_time, save_img_time, rename_time
        # return orange_time, white_time, 0, 0, 0, 0
    


