import os
import cv2
from typing import Dict, List
from io_utils.print_log import log

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(path: str, frame):
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, frame)


def del_image(imgs_save, pallet_seq, frame_count_up_no_pallet):
    if len(imgs_save[pallet_seq]) > 0:
        file_path = imgs_save[pallet_seq][0]
        if os.path.exists(file_path) and frame_count_up_no_pallet > 0:
            os.remove(file_path)
            imgs_save[pallet_seq].pop(0)
            frame_count_up_no_pallet -= 1
            # log("DEBUG", f"Del: {imgs_save[pallet_seq]}")
        
    if frame_count_up_no_pallet > 0:
        return True, frame_count_up_no_pallet
    
    frame_count_up_no_pallet = 2
    return False, frame_count_up_no_pallet


def rotate_queue(imgs_save: Dict[int, List[str]]):
    # xóa key 0, dồn các key về -1
    if len(imgs_save) > 1:
        del imgs_save[0]
        return {k-1: v for k, v in imgs_save.items()}
    else:
        imgs_save[0] = []
        return imgs_save


    