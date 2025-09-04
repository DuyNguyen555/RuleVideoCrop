import os
import cv2
import time
import config
from typing import Dict, List, Tuple
from io_utils.file_ops import save_image

class Snapshot:
    def __init__(self):
        self.check_size = False
        self.is_snap = False
        self.snapshot_frame_index = 0
        self.is_pallet_top = False
        self.frame_top_up = 0
        self.frame_top_down = 0
        self.departure_snap = False
        self.is_pallet_rule0_up = False
        self.before_motion = ""
            
    def thresholds(self, resized_h, state):
        state.y_threshold_up_above   += resized_h * 0.1
        state.y_threshold_up_below   += resized_h * 0.3
        # state.y_threshold_between    += resized_h * 0.5
        # state.y_threshold_down_above += resized_h * 0.8
        # state.y_threshold_down_below += resized_h * 1.0
        

    def save_snapshot(self, origin_frame,
                    frame_index: str,
                    output_img_dir: str,
                    imgs_save: Dict[int, List[str]],
                    pallet_seq: int) -> Tuple[int, Dict[int, List[str]], str]:
        """
        Trả về (pallet_seq_moi, imgs_save_moi, path_ảnh_vừa_lưu)
        - Tự khởi tạo key nếu chưa tồn tại
        - Nếu ngăn hiện tại đã đủ 2 ảnh thì nhảy sang ngăn mới
        """
        # đảm bảo key hiện tại tồn tại
        if pallet_seq not in imgs_save:
            imgs_save[pallet_seq] = []

        # nếu ngăn đã đủ 2 ảnh, tạo ngăn mới
        if len(imgs_save[pallet_seq]) >= 2:
            pallet_seq += 1
            if pallet_seq not in imgs_save:
                imgs_save[pallet_seq] = []

        name_img = os.path.join(output_img_dir, f"{frame_index}.jpeg")
        save_image(name_img, origin_frame)
        imgs_save[pallet_seq].append(name_img)

        # return pallet_seq, imgs_save, name_img

    def snapshot_video(self, frame, index_frame, resized_h, y_bar, state):
        t0 = time.perf_counter()
        if not self.check_size:
            self.thresholds(resized_h, state)
            self.check_size = True
        
        if self.before_motion != state.motion_current and state.motion_current in ("Down", "Up"):
            self.before_motion = state.motion_current
            self.departure_snap = False

        # print(self.departure_snap)

        # Rule cam khởi đầu chạy
        if not self.departure_snap:
            if state.motion_current == "Down":
                if self.frame_top_down == 0:
                    self.save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_0_0",
                        state.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                    self.frame_top_down += 1
                    return time.perf_counter() - t0
                else:
                    self.frame_top_down += 1
                    if self.frame_top_down >= 40:
                        self.save_snapshot(
                            frame, f"{index_frame}_{state.motion_current}_0_1",
                            state.OUTPUT_IMG, 
                            state.imgs_save,
                            state.pallet_seq
                        )
                        self.departure_snap = True
                        self.frame_top_down = 0
                        return time.perf_counter() - t0
                    
            if state.motion_current == "Up" and not self.is_pallet_rule0_up:
                self.save_snapshot(
                    frame, f"{index_frame}_{state.motion_current}_0",
                    state.OUTPUT_IMG, 
                    state.imgs_save,
                    state.pallet_seq
                )
                self.is_pallet_rule0_up = True
                return time.perf_counter() - t0

        # Rule cam đang chạy
        if y_bar == 0.0:
            if state.motion_current == "Down":
                if self.is_snap:
                    self.snapshot_frame_index += 1
                    if self.snapshot_frame_index >= 20:
                        self.save_snapshot(
                            frame, f"{index_frame}_{state.motion_current}_2",
                            state.OUTPUT_IMG, 
                            state.imgs_save,
                            state.pallet_seq
                        )
                        self.is_snap = False
                        self.snapshot_frame_index = 0
                        return time.perf_counter() - t0

            elif state.motion_current == "Up":
                if self.is_snap:
                    self.snapshot_frame_index += 1
                    if self.snapshot_frame_index >= 25:
                        self.save_snapshot(
                            frame, f"{index_frame}_{state.motion_current}_2",
                            state.OUTPUT_IMG, 
                            state.imgs_save,
                            state.pallet_seq
                        )
                        self.is_snap = False
                        self.snapshot_frame_index = 0
                        self.is_pallet_top = True
                        return time.perf_counter() - t0

                elif self.is_pallet_top:
                    self.frame_top_up += 1
                    if self.frame_top_up >= 50:
                        self.save_snapshot(
                            frame, f"{index_frame}_{state.motion_current}_3",
                            state.OUTPUT_IMG, 
                            state.imgs_save,
                            state.pallet_seq
                        )
                        self.is_pallet_top = False
                        self.frame_top_up = 0
                        return time.perf_counter() - t0

        else:
            self.frame_top_up = 0

        if state.y_threshold_up_above < y_bar < state.y_threshold_up_below:
            if not self.is_snap:
                self.save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_1",
                        state.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                self.is_snap = True
                return time.perf_counter() - t0
        
        return time.perf_counter() - t0