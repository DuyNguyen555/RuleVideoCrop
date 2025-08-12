import cv2
import config
import numpy as np
from pyzbar import pyzbar
import os

def RoiCropped(frame, height, width):
    x1 = int(width  * config.x1_rel)
    x2 = int(width  * config.x2_rel)
    y1 = int(height * config.y1_rel)
    y2 = int(height * config.y2_rel)
    return frame[y1:y2, x1:x2]

def DetectOrangeBar(frame, kernel, frame_bgr=None, y_threshold=10):
    lower, upper = np.array([10, 150, 210]), np.array([16, 255, 255])
    mask_roi = cv2.inRange(frame, lower, upper)

    
    mask_clean_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("Mask", mask_clean_roi)

    contours, _ = cv2.findContours(mask_clean_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 50:
            x, y, w, h = cv2.boundingRect(cnt)
            y_center = y + h / 2
            boxes.append((x, y, w, h, y_center))

    merged_boxes = []
    boxes.sort(key=lambda b: b[4])
    while boxes:
        base = boxes.pop(0)
        group = [base]
        for b in boxes[:]:
            if abs(b[4] - base[4]) <= y_threshold:
                group.append(b)
                boxes.remove(b)
        min_y = min(g[1] for g in group)
        max_y = max(g[1] + g[3] for g in group)
        merged_boxes.append((0, min_y, frame.shape[1], max_y - min_y))

    # filtered_mask = np.zeros_like(mask_clean_roi)
    bar_box_points = None
    if merged_boxes:
        # Lấy box lớn nhất
        x, y, w, h = max(merged_boxes, key=lambda b: b[3])
        # cv2.rectangle(filtered_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        bar_box_points = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float16)

    # result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=filtered_mask)
    bar_found = bar_box_points is not None
    # cropped = None

    # if bar_found:
    #     # Dùng bounding box để crop ảnh gốc
    #     x, y, w, h = cv2.boundingRect(bar_box_points.astype(int))
    #     cropped = frame_bgr[y:y+h, x:x+w]
    #     cv2.imshow("Re:", cropped)


    return bar_found, bar_box_points

def DetectQRCode(frame, resized_box_points, resize_factor):
    """
    Phát hiện QR code trong frame gốc từ MỘT vùng tọa độ cho trước.
    Tọa độ đầu vào (resized_box_points) là của vùng ROI đã được resize và crop.
    """
    if resized_box_points is None or len(resized_box_points) != 4:
        return False, []

    original_height, original_width = frame.shape[:2]

    # BƯỚC 1: TÍNH TOÁN CÁC THÔNG SỐ BIẾN ĐỔI 
    # Kích thước của frame sau khi resize
    resized_width = original_width // resize_factor
    resized_height = original_height // resize_factor

    # Tọa độ (offset) của vùng ROI so với frame ĐÃ RESIZE
    x_offset_in_resized = int(resized_width * config.x1_rel)
    y_offset_in_resized = int(resized_height * config.y1_rel)

    # BƯỚC 2: QUY ĐỔI TỌA ĐỘ NGƯỢC LẠI
    original_box_points = []
    for point_in_roi in resized_box_points:
        # point_in_roi[0] là x_roi, point_in_roi[1] là y_roi
        
        # Chuyển từ tọa độ ROI sang tọa độ của frame đã resize
        x_in_resized = point_in_roi[0] + x_offset_in_resized
        y_in_resized = point_in_roi[1] + y_offset_in_resized

        # Chuyển từ tọa độ frame đã resize về tọa độ frame gốc
        x_in_original = x_in_resized * resize_factor
        y_in_original = y_in_resized * resize_factor
        
        original_box_points.append([x_in_original, y_in_original])
    
    original_box_points = np.array(original_box_points, dtype=np.float32)

    (tl, tr, br, bl) = original_box_points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Kiểm tra khi kích thước bằng 0
    if maxWidth <= 0 or maxHeight <= 0:
        return False, []

    dst_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(original_box_points, dst_points)
    warped_qr_area = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    gray_qr_area = cv2.cvtColor(warped_qr_area, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('qr', gray_qr_area)

    # Xu ly QR
    decoded_objects = pyzbar.decode(gray_qr_area)
    
    results = []
    isQR = False
    if decoded_objects:
        isQR = True
        for obj in decoded_objects:
            results.append(obj.data.decode("utf-8"))
    return isQR, results

def Motion(frame, frame_index, bar_found, bar_box_points, roi, motion_detector, isQR):
    direction = motion_detector.update(frame, isQR)
    # print("Direction:", direction)

    if config.motion_before == "" or config.motion_current == "":
        config.motion_before  = direction
        config.motion_current = direction
    elif direction != config.motion_current:
        config.motion_changeframe = True
        config.motion_before = config.motion_current
        config.motion_current = direction


    config.trigger_start = False
    config.trigger_stop = False

    # Logic khi đi xuống
    if config.motion_before in ["Stop", "Right"] and config.motion_current == "Down":
        config.name_record = frame_index
        config.trigger_start = True

    if config.motion_current == "Down" and bar_found:
        center_y = np.mean(bar_box_points[:, 1])
        roi_height = roi.shape[0]
        y_threshold_down = roi_height * 0.5
        y_threshold_down2 = roi_height * 0.45

        if y_threshold_down2 <= center_y <= y_threshold_down:
            config.trigger_start = False
            config.trigger_stop = True
        
        # print(f"Frame{frame_index} Y_center: {center_y}")
        # print(y_threshold_down)

    if config.motion_before in ["Stop", "Down"] and config.motion_current in ("Left", "Right"):
        config.trigger_start = False
        config.trigger_stop = True

    
    # Logic khi đi lên
    if config.motion_before in ["Stop", "Left", "Right"] and config.motion_current == "Up":
        config.name_record = frame_index
        config.trigger_start = True

    if config.motion_current == "Up" and bar_found:
        center_y = np.mean(bar_box_points[:, 1])
        roi_height = roi.shape[0]
        y_threshold_down = roi_height * 0.6
        y_threshold_down2 = roi_height * 0.65

        if y_threshold_down <= center_y <= y_threshold_down2:
            config.trigger_start = False
            config.trigger_stop = True
        
        # print(f"Frame{frame_index} Y_center: {center_y}")
        # print(y_threshold_down)

    if config.motion_before in ["Stop", "Up"] and config.motion_current in ("Left", "Right"):
        config.trigger_start = False
        config.trigger_stop = True
        
        # print(f"Frame{frame_index} Y_center: {center_y}")
        # print(y_threshold_down)


def RenameVideo(len_qr):
    # if len_qr > 0 and config.len_record > 0:
    if config.len_record > 0:
        try:
            index = 0
            config.frame_waitting_qr += 1
            if config.motion_current == "Up" or config.motion_before == "Up":
                # print(1)
                index = -1
            elif config.motion_current == "Down" or config.motion_before == "Down":
                # print(2)
                index = 0
            else:
                # print(3)
                index = 0

            # Rule logic QR
            if len_qr > 0:
                old_path = config.name_record_save.pop(index)
                qr_code = config.ls_qr[index]
                new_path = os.path.join(config.output_dir, f"{qr_code}.mp4")

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    config.name_video_saved.append(config.ls_qr.pop(index))
                    config.len_record -= 1
                    config.frame_waitting_qr = 0

                    print(f" Done:\n  {old_path}\n→ {new_path}")
                else:
                    print(f"No file: {old_path}")
            else:
                if config.frame_waitting_qr >= config.threshold_waitting_qr:
                    old_path = config.name_record_save.pop(index)
                    basename = os.path.basename(old_path)
                    config.name_video_saved.append(basename) 
                    config.len_record -= 1
                    config.frame_waitting_qr = 0

        except Exception as e:
            print(f"Error when change name: {e}")
