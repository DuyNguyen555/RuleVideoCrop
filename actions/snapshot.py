import os
import cv2
import time
import config
from typing import Dict, List, Tuple
from io_utils.file_ops import save_image
from vision.motion import update_direction

check_size = False
count_below = 0
count_up = 0
is_snap = False
cooldown = 0
is_motion = False
directory = None
prev_y_bar = None
is_pallet_top = False
cooldown_top_up = 0
cooldown_top_down = 0
departure_snap = False
is_pallet_rule0_up = False
before_motion = ""


def thresholds(resized_h, state):
    state.y_threshold_up_above   += resized_h * 0.1
    state.y_threshold_up_below   += resized_h * 0.3
    state.y_threshold_between    += resized_h * 0.5
    state.y_threshold_down_above += resized_h * 0.8
    state.y_threshold_down_below += resized_h * 1.0
    

def save_snapshot(origin_frame,
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

def snapshot_video(frame, index_frame, resized_h, y_bar, state):
    t0 = time.perf_counter()
    global check_size, count_below, is_snap, cooldown, directory, prev_y_bar, is_pallet_top, cooldown_top_up, cooldown_top_down, departure_snap, is_pallet_rule0_up, before_motion
    if not check_size:
        thresholds(resized_h, state)
        check_size = True
    
    if before_motion != state.motion_current and state.motion_current in ("Down", "Up"):
        before_motion = state.motion_current
        departure_snap = False

    # print(departure_snap)
    # Rule cam khởi đầu chạy
    if not departure_snap:
        if state.motion_current == "Down":
            if cooldown_top_down == 0:
                save_snapshot(
                    frame, f"{index_frame}_{state.motion_current}_0_0",
                    config.OUTPUT_IMG, 
                    state.imgs_save,
                    state.pallet_seq
                )
                cooldown_top_down += 1
                return time.perf_counter() - t0
            else:
                cooldown_top_down += 1
                if cooldown_top_down >= 40:
                    save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_0_1",
                        config.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                    departure_snap = True
                    cooldown_top_down = 0
                    return time.perf_counter() - t0
                
        if state.motion_current == "Up" and not is_pallet_rule0_up:
            save_snapshot(
                frame, f"{index_frame}_{state.motion_current}_0",
                config.OUTPUT_IMG, 
                state.imgs_save,
                state.pallet_seq
            )
            is_pallet_rule0_up = True
            return time.perf_counter() - t0

    # Rule cam đang chạy
    if y_bar == 0.0:
        if state.motion_current == "Down":
            if is_snap:
                cooldown += 1
                if cooldown >= 20:
                    save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_2",
                        config.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                    is_snap = False
                    cooldown = 0
                    return time.perf_counter() - t0

        elif state.motion_current == "Up":
            if is_snap:
                cooldown += 1
                if cooldown >= 25:
                    save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_2",
                        config.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                    is_snap = False
                    cooldown = 0
                    is_pallet_top = True
                    return time.perf_counter() - t0

            elif is_pallet_top:
                # print(cooldown_top_up)
                cooldown_top_up += 1
                if cooldown_top_up >= 50:
                    save_snapshot(
                        frame, f"{index_frame}_{state.motion_current}_3",
                        config.OUTPUT_IMG, 
                        state.imgs_save,
                        state.pallet_seq
                    )
                    is_pallet_top = False
                    cooldown_top_up = 0
                    return time.perf_counter() - t0


    else:
        cooldown_top_up = 0
        # directory, prev_y_bar = update_direction(y_bar, directory, prev_y_bar)
        # print(directory)
        # pass

    if state.y_threshold_up_above < y_bar < state.y_threshold_up_below:
        if not is_snap:
            save_snapshot(
                    frame, f"{index_frame}_{state.motion_current}_1",
                    config.OUTPUT_IMG, 
                    state.imgs_save,
                    state.pallet_seq
                )
            is_snap = True
            return time.perf_counter() - t0
    
    return time.perf_counter() - t0