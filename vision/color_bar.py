import cv2
import numpy as np
from typing import Tuple, Optional
import config
from .geometry import resize_points_to_roi
import time

def detect_orange_bar(hsv, kernel, show_result=True):
    t0 = time.perf_counter()
    lower = np.array(config.ORANGE_LOWER, dtype=np.uint8)
    upper = np.array(config.ORANGE_UPPER, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Mask",mask)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 100:
            x, y, w, h = cv2.boundingRect(cnt)
            y_center = y + h / 2
            boxes.append((x, y, w, h, y_center))

    # gộp theo hàng ngang
    merged = []
    boxes.sort(key=lambda b: b[4])
    y_threshold = max(1, 10)
    while boxes:
        base = boxes.pop(0)
        group = [base]
        for b in boxes[:]:
            if abs(b[4] - base[4]) <= y_threshold:
                group.append(b)
                boxes.remove(b)
        min_y = min(g[1] for g in group)
        max_y = max(g[1] + g[3] for g in group)
        merged.append((0, min_y, hsv.shape[1], max_y - min_y))

    if not merged:
        t1 = time.perf_counter()
        return None, 0.0, time.perf_counter() - t0

    # chọn box cao nhất
    x, y, w, h = max(merged, key=lambda b: b[3])
    bar_box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    y_center = (bar_box[0, 1] + bar_box[2, 1]) / 2

    filled = hsv.copy()
    # Fill vùng mask bên trong bar_box
    mask_filled = np.zeros_like(mask_clean)
    cv2.fillPoly(mask_filled, [bar_box], 255)

    # Giữ lại chỉ phần trong bar_box
    filled_result = cv2.bitwise_and(filled, filled, mask=mask_filled)

    if show_result:
        cv2.polylines(filled_result, [bar_box], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(filled_result, f"y_center={int(y_center)}", (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Orange Bar Detection", filled_result)

    return filled_result, y_center, time.perf_counter() - t0


def detect_white(hsv, roi, show_result=True):
    t0 = time.perf_counter()
    if hsv is None:
        return [], [], time.perf_counter() - t0

    mask = cv2.inRange(hsv, np.array(config.WHITE_LOWER, np.uint8), np.array(config.WHITE_UPPER, np.uint8))
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    crops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # lọc nhiễu nhỏ
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Lọc theo kích thước tối thiểu
        if w < 30 or h < 10:
            continue

        # Lọc theo tỉ lệ width/height (label thường nằm ngang)
        aspect_ratio = w / float(h)
        if aspect_ratio < 2.0 or aspect_ratio > 10.0:
            continue

        boxes.append((x, y, w, h))

    boxes_np = np.array(boxes, dtype=np.float32).reshape(-1, 4)

    if boxes_np.size == 0:
        return crops, [], time.perf_counter() - t0
    
    for (x, y, w, h) in boxes_np:
        # Lấy 4 điểm góc của box
        pts = np.array([[x, y],
                        [x + w, y],
                        [x, y + h],
                        [x + w, y + h]], dtype=np.float32)

        # Resize về ROI gốc
        pts_resized = resize_points_to_roi(
            pts,
            roi_shape=roi.shape,
            roi_resized_wh=(hsv.shape[1], hsv.shape[0]),
            normalized=False
        ).astype(np.int32)

        # Lấy bounding rect cho box sau resize
        x0, y0, ww, hh = cv2.boundingRect(pts_resized)

        H, W = roi.shape[:2]
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(W, x0 + ww), min(H, y0 + hh)

        if x2 > x1 and y2 > y1:
            cropped = roi[y1:y2, x1:x2]
            crops.append(cropped)

    if show_result:
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, f"{w}x{h}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("White Detection", result)
        cv2.imshow("White Mask Clean", mask_clean)
        for i, c in enumerate(crops): cv2.imshow(f"crop_{i}", c)

    return crops, boxes, time.perf_counter() - t0



# def detect_orange_bar(hsv_roi_resized, kernel, roi_original_shape, roi_resized_wh, base_frame):
    # lower = np.array(config.ORANGE_LOWER, dtype=np.uint8)
    # upper = np.array(config.ORANGE_UPPER, dtype=np.uint8)

    # mask = cv2.inRange(hsv_roi_resized, lower, upper)
    # # mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("hsv", hsv_roi_resized)
    # cv2.imshow("Mask", mask)

    # contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # boxes = []
    # for cnt in contours:
    #     if cv2.contourArea(cnt) >= 100:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         y_center = y + h/2
    #         boxes.append((x, y, w, h, y_center))

    # # gộp theo hàng ngang
    # merged = []
    # boxes.sort(key=lambda b: b[4])
    # y_threshold = max(1, 10)  # bạn có thể truyền vào ngoài nếu muốn
    # while boxes:
    #     base = boxes.pop(0)
    #     group = [base]
    #     for b in boxes[:]:
    #         if abs(b[4] - base[4]) <= y_threshold:
    #             group.append(b); boxes.remove(b)
    #     min_y = min(g[1] for g in group)
    #     max_y = max(g[1] + g[3] for g in group)
    #     merged.append((0, min_y, hsv_roi_resized.shape[1], max_y - min_y))

    # if not merged:
    #     return False, 0.0, None

    # x, y, w, h = max(merged, key=lambda b: b[3])
    # bar_box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    # y_center = (bar_box[0,1] + bar_box[2,1]) / 2


    # pts = resize_points_to_roi(
    #     bar_box,
    #     roi_shape=roi_original_shape,
    #     roi_resized_wh=roi_resized_wh,
    #     normalized=False
    # ).astype(np.int32)

    # x0, y0, ww, hh = cv2.boundingRect(pts)
    # H, W = base_frame.shape[:2]
    # x1, y1 = max(0, x0), max(0, y0)
    # x2, y2 = min(W, x0 + ww), min(H, y0 + hh)
    # cropped = base_frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

    # return True, y_center, cropped