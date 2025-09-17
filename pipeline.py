import cv2
import time
import config
from state import State

from vision.color_bar import detect_orange_bar, detect_white
from vision.qr import QRDetector
from vision.geometry import roi_bounds
from vision.motion import MotionDetector
from actions.snapshot import Snapshot
from actions.name_video import NameVideo

class Pipeline:
    def __init__(self, state: State, motion_detector: MotionDetector, qr_detector: QRDetector, snapshot: Snapshot, logic_name_video: NameVideo):
        self.state = state
        self.motion_detector = motion_detector
        self.qr_detector = qr_detector
        self.snapshot = snapshot
        self.logic_name_video = logic_name_video


    def reset_when_motion_change(self, state):
        if state.motion_current != state.motion_before:
            state.motion_before = state.motion_current
        
        if state.motion_before in ("Down", "Up") and state.motion_current in ("None", "Left", "Right"):
            state.ls_qr = []
            state.imgs_save = {0: []}
            state.pallet_seq = 0
            state.name_video_saved = []
            return state.ls_qr, state.imgs_save, state.pallet_seq, state.name_video_saved, state.motion_before
        else:
            return state.ls_qr, state.imgs_save, state.pallet_seq, state.name_video_saved, state.motion_before


    def process_frame(self, frame, kernel, frame_index: int):

        # Rule ROI
        if not self.state.roi_ready:
            self.state.roi_x1, self.state.roi_y1, self.state.roi_x2, self.state.roi_y2 = roi_bounds(frame_shape=frame.shape)
            self.state.roi_ready = True
        
        roi = frame[self.state.roi_y1:self.state.roi_y2, self.state.roi_x1:self.state.roi_x2]

        resized = cv2.resize(roi, (roi.shape[1] // config.RESIZE_FACTOR,
                                   roi.shape[0] // config.RESIZE_FACTOR),
                                    interpolation=cv2.INTER_LINEAR)
        
        # cv2.imwrite("resize.png", resized)
        hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # self._init_thresholds(resized.shape[0])

        # Rule Orange bar
        frame_bar, y_bar, orange_time = detect_orange_bar(hsv=hsv_resized,
                                                      kernel=kernel,
                                                      show_result=False)
                
        # Rule White in Orange bar
        white_frame, boxes, white_time = detect_white(frame_bar, roi, show_result=False)
        has_white = bool(boxes)
        # print(has_white)

        # Rule QRcode
        qr_time = self.qr_detector.detect_qr(frames=white_frame, 
                                            state=self.state)

        # Rule Motion
        self.state.motion_current, self.state.departure, motion_time = self.motion_detector.find_motion(roi=resized,
                                                                                                        y_bar=y_bar,
                                                                                                        state=self.state)
        self.state.ls_qr, self.state.imgs_save, self.state.pallet_seq, self.state.name_video_saved, self.state.motion_before = self.reset_when_motion_change(self.state)
        # print(self.state.motion_current)

        # Rule Snapshot
        save_img_time = self.snapshot.snapshot_video(frame=frame,
                                                    index_frame=frame_index,
                                                    resized_h=resized.shape[0],
                                                    y_bar=y_bar,
                                                    has_white=has_white,
                                                    state=self.state)
        
        # print(self.state.imgs_save)

        # Rule del link name
        self.logic_name_video.remove_link_img(self.state)

        # # Rule Rename
        self.state.imgs_save, self.state.ls_qr, self.state.name_video_saved, self.state.pallet_seq, rename_time = self.logic_name_video._rename_if_ready(state=self.state)

        # print(self.state.motion_current)
        # print(self.state.ls_qr)
        return orange_time, white_time, qr_time, motion_time, save_img_time, rename_time    


