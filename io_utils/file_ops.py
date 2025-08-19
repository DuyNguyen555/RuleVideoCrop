import os
import cv2
from typing import Dict, List

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_image(path: str, frame):
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, frame)

def rotate_queue(imgs_save: Dict[int, List[str]]):
    # xóa key 0, dồn các key về -1
    if len(imgs_save) > 1:
        del imgs_save[0]
        return {k-1: v for k, v in imgs_save.items()}
    else:
        imgs_save[0] = []
        return imgs_save
