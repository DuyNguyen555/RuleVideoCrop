import cv2
import numpy as np
from pyzbar import pyzbar

def scan_pyzbar(gray_or_bgr_frame):
    # chấp nhận input gray hoặc bgr
    if len(gray_or_bgr_frame.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr_frame
    gray = np.ascontiguousarray(gray)

    # gray_small = cv2.resize(
    #     gray,
    #     (gray.shape[1] // 4, gray.shape[0] // 4),
    #     interpolation=cv2.INTER_AREA  
    # )
    # cv2.imshow("Gray", gray_small)

    res = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
    return [b.data.decode("utf-8") for b in res] or []

def detect_qr(frame_bgr, allow_filter=None):
    blur = cv2.GaussianBlur(frame_bgr, (3,3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    codes = scan_pyzbar(gray)
    # print("QRCode: ", codes)
    if allow_filter:
        codes = [c for c in codes if allow_filter(c)]
    return codes
