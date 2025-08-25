import re
import cv2
import time
import numpy as np
from pyzbar import pyzbar


_QR_PATTERN = re.compile(r"^[A-Z]{2}-\d{3}-\d$")

def check_qr(qr):
    """
    Kiểm tra mã QR: [2 chữ]-[3 số]-[1 số]
    """
    return bool(_QR_PATTERN.match(qr))

def scan_pyzbar(gray_or_bgr_frame, show_result):
    # chấp nhận input gray hoặc bgr
    if len(gray_or_bgr_frame.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr_frame
    gray = np.ascontiguousarray(gray)

    if show_result:
        gray_small = cv2.resize(
            gray,
            (gray.shape[1] // 4, gray.shape[0] // 4),
            interpolation=cv2.INTER_AREA  
        )
        cv2.imshow("Gray", gray_small)

    res = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
    return [b.data.decode("utf-8") for b in res] or []

def detect_qr(frames, ls_qr, name_video_saved, show_result=False):
    t0 = time.perf_counter()

    for frame in frames:
        blur = cv2.GaussianBlur(frame, (3,3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        found_codes = scan_pyzbar(gray, show_result)
        for qr in found_codes:
            if check_qr(qr) and qr not in ls_qr and qr not in name_video_saved:
                ls_qr.append(qr)

    if show_result:
        print("QRCode: ", ls_qr)

    return time.perf_counter() - t0
