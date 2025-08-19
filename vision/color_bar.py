import cv2
import numpy as np
from typing import Tuple, Optional
import config
from .geometry import resize_points_to_roi

def detect_orange_bar(hsv_roi_resized, kernel, roi_original_shape, roi_resized_wh, base_frame):
    lower = np.array(config.ORANGE_LOWER, dtype=np.uint8)
    upper = np.array(config.ORANGE_UPPER, dtype=np.uint8)

    mask = cv2.inRange(hsv_roi_resized, lower, upper)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Mask", mask_clean)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 100:
            x, y, w, h = cv2.boundingRect(cnt)
            y_center = y + h/2
            boxes.append((x, y, w, h, y_center))

    # gộp theo hàng ngang
    merged = []
    boxes.sort(key=lambda b: b[4])
    y_threshold = max(1, 10)  # bạn có thể truyền vào ngoài nếu muốn
    while boxes:
        base = boxes.pop(0)
        group = [base]
        for b in boxes[:]:
            if abs(b[4] - base[4]) <= y_threshold:
                group.append(b); boxes.remove(b)
        min_y = min(g[1] for g in group)
        max_y = max(g[1] + g[3] for g in group)
        merged.append((0, min_y, hsv_roi_resized.shape[1], max_y - min_y))

    if not merged:
        return False, 0.0, None

    x, y, w, h = max(merged, key=lambda b: b[3])
    bar_box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    y_center = (bar_box[0,1] + bar_box[2,1]) / 2


    pts = resize_points_to_roi(
        bar_box,
        roi_shape=roi_original_shape,
        roi_resized_wh=roi_resized_wh,
        normalized=False
    ).astype(np.int32)

    x0, y0, ww, hh = cv2.boundingRect(pts)
    H, W = base_frame.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(W, x0 + ww), min(H, y0 + hh)
    cropped = base_frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

    return True, y_center, cropped