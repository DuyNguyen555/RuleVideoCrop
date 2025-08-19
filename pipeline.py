import cv2
import math
import numpy as np
from typing import Tuple
import config
from state import State

from vision.color_bar import detect_orange_bar
from vision.qr import detect_qr
from vision.motion import MotionDetector
from vision.geometry import check_qr
from vision.draw import draw_thresholds
from actions.snapshot import save_snapshot
from actions.renamer import rename_pair_from_queue

class Pipeline:
    def __init__(self, state: State, motion_detector: MotionDetector):
        self.state = state
        self.motion_detector = motion_detector

    def _init_roi_bounds(self, frame_shape):
        if self.state.roi_ready:
            return
        H, W = frame_shape[:2]

        # clamp
        x1r = max(0.0, min(1.0, float(getattr(config, "X1_REL", 0.0))))
        x2r = max(0.0, min(1.0, float(getattr(config, "X2_REL", 1.0))))
        y1r = max(0.0, min(1.0, float(getattr(config, "Y1_REL", 0.0))))
        y2r = max(0.0, min(1.0, float(getattr(config, "Y2_REL", 1.0))))

        # nếu cấu hình ngược → full frame
        if x2r <= x1r: x1r, x2r = 0.0, 1.0
        if y2r <= y1r: y1r, y2r = 0.0, 1.0

        # start dùng floor, end dùng ceil (end là exclusive)
        x1 = int(math.floor(x1r * W))
        x2 = int(math.ceil (x2r * W))
        y1 = int(math.floor(y1r * H))
        y2 = int(math.ceil (y2r * H))

        # kẹp biên (cho phép end == W/H vì slice end là exclusive)
        x1 = max(0, min(x1, W-1))
        y1 = max(0, min(y1, H-1))
        x2 = max(x1+1, min(x2, W))
        y2 = max(y1+1, min(y2, H))

        self.state.roi_x1, self.state.roi_y1 = x1, y1
        self.state.roi_x2, self.state.roi_y2 = x2, y2
        self.state.roi_ready = True

    def _init_thresholds(self, resized_h: int):
        if not self.state.check_size:
            self.state.y_threshold_up_above   = resized_h * 0.01
            self.state.y_threshold_up_below   = resized_h * 0.2
            self.state.y_threshold_between    = resized_h * 0.5
            self.state.y_threshold_down_above = resized_h * 0.9
            self.state.y_threshold_down_below = resized_h * 1.0
            self.state.check_size = True

    def _sort_qr_motion(self, motion):
        if motion == "Down":
            return True
        elif motion == "Up":
            return False
        
    def _qr_base(self, qr: str) -> str:
        return qr[:-1]

    def _has_tail(self, base: str, tail: str) -> bool:
        return any(s.startswith(base) and s.endswith(tail) for s in self.state.ls_qr)

    def _ingest_qr(self, codes, sort):
        """
        Kiểm tra điều kiện qr hợp lệ
        nếu qr kết thúc bằng '1' hoặc '2' và count_qr_1_2 < 1: tăng đếm + append
        ngược lại: scanned_qr = True; reset đếm; append
        """
        for qr in codes:
            if not check_qr(qr):
                continue
            if qr in self.state.ls_qr or qr in self.state.name_video_saved:
                continue

            if qr.endswith(('1', '2')) and self.state.count_qr_1_2 < 1:
                self.state.count_qr_1_2 += 1
                self.state.ls_qr.append(qr)
                base = self._qr_base(qr)
                tail = qr[-1]

                # Nếu đang Up & nhận '2' trước '1' -> đợi '1'
                if self.state.motion_current == "Up" and tail == "2" and not self._has_tail(base, "1"):
                    self.state.wait_pair_active = True
                    self.state.wait_base = base
                    self.state.wait_expected_tail = "1"
                    self.state.wait_counter = 0

                # Nếu đang Down & nhận '1' trước '2' -> đợi '2'
                if self.state.motion_current == "Down" and tail == "1" and not self._has_tail(base, "2"):
                    self.state.wait_pair_active = True
                    self.state.wait_base = base
                    self.state.wait_expected_tail = "2"
                    self.state.wait_counter = 0

                # Nếu nhận được đúng “cặp” trong lúc đang đợi -> tắt chờ
                if self.state.wait_pair_active and base == self.state.wait_base and tail == self.state.wait_expected_tail:
                    self.state.wait_pair_active = False
                    self.state.wait_base = ""
                    self.state.wait_expected_tail = ""
                    self.state.wait_counter = 0
            else:
                self.state.scanned_qr = True
                self.state.count_qr_1_2 = 0
                self.state.ls_qr.append(qr)

        if len(self.state.ls_qr) > 1:
            self.state.ls_qr.sort(key=lambda x: int(x[-1]), reverse = sort)
            # print("===Sort: ", self.state.ls_qr)

    def process_frame(self, frame, kernel, frame_index: int):
        if frame_index % 5 != 0:
            return 0

        self._init_roi_bounds(frame.shape)
        x1,y1,x2,y2 = self.state.roi_x1, self.state.roi_y1, self.state.roi_x2, self.state.roi_y2
        roi = frame[y1:y2, x1:x2]

        resized = cv2.resize(roi, (roi.shape[1] // config.RESIZE_FACTOR,
                                   roi.shape[0] // config.RESIZE_FACTOR),
                                    interpolation=cv2.INTER_LINEAR)
        hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        self._init_thresholds(resized.shape[0])

        # cv2.imshow("Re", resized)

        # Orange bar
        found, bar_p, bar_crop = detect_orange_bar(
            hsv_roi_resized=hsv_resized,
            kernel=kernel,
            roi_original_shape=roi.shape,
            roi_resized_wh=(resized.shape[1], resized.shape[0]),
            base_frame=roi
        )
        bar_found_this_frame = bool(found)

        # Frame hiện tại có QR không 
        qr_found_this_frame = False

        # print(found)
        if found and not self.state.scanned_qr and bar_crop is not None:
            codes = detect_qr(bar_crop)
            if codes:
                qr_found_this_frame = True
                self._ingest_qr(codes, self.state.sort)

        # scan cooldown khi đã có qr
        if self.state.scanned_qr:
            self.state.frame_stop_scan -= 1
            if self.state.frame_stop_scan <= 0:
                self.state.scanned_qr = False
                self.state.frame_stop_scan = config.FRAME_STOP_SCAN

        # vis = draw_thresholds(resized.copy(), (self.state.y_threshold_up_above, self.state.y_threshold_up_below), bar_p)
        # cv2.imshow("Vis", vis)

        # Motion logic
        if not self.state.departure:
            direction = self.motion_detector.update(roi)
            self.state.sort = self._sort_qr_motion(direction)
            # print(direction)
            # if direction in ("Up", "Down"):
            if direction == "Down":
                self.state.count_crop_img_qr += 1
                self.state.crop_next = True
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_down_0", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.motion_current = direction
                self.state.departure = True

            elif direction == "Up":
                self.state.count_crop_img_qr += 1
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_up_0", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.motion_current = direction
                self.state.departure = True

        # Nếu không phát hiện qr trong số frame chỉ định thì reset hướng
        else:
            if qr_found_this_frame:
                self.state.directory_cooldown_frame = 0
                # print(self.state.directory_cooldown_frame)
            else:
                self.state.directory_cooldown_frame += 1
                # print(self.state.directory_cooldown_frame)
                if self.state.directory_cooldown_frame >= config.DIRECTORY_COOLDOWN and len(self.state.imgs_save[0]) == 0:
                    self.state.crop_2 = 0
                    self.state.departure = False
                    self.state.motion_current = ""
                    self.state.directory_cooldown_frame = 0
                    self.state.count_qr_1_2 = 0
                    self.motion_detector.reset_direction()
                    self.state.up_no_bar_counter = 0
                    self.state.up_no_bar_captured = False
            
        # tracked_points = self.motion_detector.p0
        # if tracked_points is not None:
        #     for point in tracked_points:
        #         x, y = point.ravel()
        #         cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1) 

        # vis = draw_thresholds(resized.copy(), (self.state.y_threshold_up_above, self.state.y_threshold_up_below), bar_p)

        # if tracked_points is not None:
        #     for point in tracked_points:
        #         x, y = point.ravel()
        #         cv2.circle(vis, (int(x / config.RESIZE_FACTOR), int(y / config.RESIZE_FACTOR)), 3, (0, 255, 0), -1)

        # cv2.imshow("Vis", vis)

        # Logic Down
        if self.state.motion_current == "Down":
            if self.state.crop_next and self.state.crop_2 >= config.CROP_2_COOLDOWN_DOWN:
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_down_2", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.crop_next = False
                self.state.crop_2 = 0

            if (self.state.y_threshold_up_above < bar_p < self.state.y_threshold_up_below) and (not self.state.crop_next):
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_down_1", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.crop_next = True

        # Logic Up
        if self.state.motion_current == "Up":
            # print(f"--=Crop Next {frame_index}:", self.state.crop_next)
            if not bar_found_this_frame and not self.state.up_no_bar_captured:
                self.state.up_no_bar_counter += 1
                if self.state.up_no_bar_counter >= config.UP_NO_BAR_MAX_FRAMES:
                    self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                        frame, f"{frame_index}_up_timeout", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                    )
                    self.state.up_no_bar_captured = True
                    # dừng chuỗi chụp tiếp theo trong phiên hiện tại
                    self.state.crop_next = False
                    self.state.crop_2 = 0
            else:
                # nếu có thấy bar thì reset đếm
                if bar_found_this_frame:
                    self.state.up_no_bar_counter = 0

            if self.state.crop_next and self.state.crop_2 >= config.CROP_2_COOLDOWN_UP:
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_up_2", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.crop_next = False
                self.state.crop_2 = 0
            if (self.state.y_threshold_up_above < bar_p < self.state.y_threshold_up_below) and (not self.state.crop_next):
                self.state.pallet_seq, self.state.imgs_save, _ = save_snapshot(
                    frame, f"{frame_index}_up_1", config.OUTPUT_IMG, self.state.imgs_save, self.state.pallet_seq
                )
                self.state.crop_next = True

        # Logic có đuôi qrcode là 1 hoặc 2
        if self.state.wait_pair_active:
            # nếu đã có đủ cặp trong ls_qr thì tắt chờ
            if self._has_tail(self.state.wait_base, self.state.wait_expected_tail):
                self.state.wait_pair_active = False
                self.state.wait_base = ""
                self.state.wait_expected_tail = ""
                self.state.wait_counter = 0
            else:
                self.state.wait_counter += 1
                if self.state.wait_counter >= config.WAIT_PAIR_FRAMES:
                    # Quá hạn -> tự tạo mã còn thiếu
                    synth = self.state.wait_base + self.state.wait_expected_tail
                    if synth not in self.state.ls_qr and synth not in self.state.name_video_saved:
                        self.state.ls_qr.append(synth)
                        # sắp xếp theo ký tự cuối (tăng/giảm theo self.state.sort)
                        if len(self.state.ls_qr) > 1:
                            self.state.ls_qr.sort(key=lambda x: int(x[-1]), reverse = self.state.sort)
                            # print("+++Sort: ", self.state.ls_qr)
                    # tắt chờ
                    self.state.wait_pair_active = False
                    self.state.wait_base = ""
                    self.state.wait_expected_tail = ""
                    self.state.wait_counter = 0

        # tick đếm
        if self.state.crop_next:
            # print(self.state.crop_2)
            self.state.crop_2 += 1

        # print(frame_index, ">>>", self.state.imgs_save)
        # print("Motion:", self.state.motion_current)

        # Rename khi đủ điều kiện
        self._rename_if_ready()

    def _rename_if_ready(self):
        if self.state.wait_pair_active:
            return
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
        if changed:
            self.state.is_rename_img = False
            self.state.time_rename_img = config.TIME_RENAME_IMG_DEFAULT