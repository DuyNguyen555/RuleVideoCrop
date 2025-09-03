import re
import cv2
import time
import numpy as np
from pyzbar import pyzbar


_QR_PATTERN = re.compile(r"^[A-Z]{2}-\d{3}-\d$")

class QRDetector:
    def __init__(self):
        self.two_qr = 0
        self.has_found_qr = False
        self.frame_stop_scan_qr = 0

    def sort_qr_motion(self, motion):
        if motion == "Down":
            return True
        else:
            return False
        
    def hasFound(self, qr):
        """
        Kiểm tra đây có phải rule QR đặc biệt không
        rule QR đặc biệt: mã QR có ký tự cuối cùng là "1" hoặc "2"
        """
        if qr.endswith("1") or qr.endswith("2"):
            self.two_qr += 1
            if self.two_qr == 2:
                self.two_qr = 0
                return True
            else:
                return False

        return True

    def check_qr(self, qr):
        """
        Kiểm tra mã QR: [2 chữ]-[3 số]-[1 số]
        """
        return bool(_QR_PATTERN.match(qr))

    def scan_pyzbar(self, gray_or_bgr_frame, show_result):
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

    def detect_qr(self, frames, ls_qr, name_video_saved, motion, show_result=False):
        t0 = time.perf_counter()

        if self.has_found_qr:
            # print(self.frame_stop_scan_qr)
            self.frame_stop_scan_qr += 1
            if self.frame_stop_scan_qr >= 21:
                self.has_found_qr = False
                self.frame_stop_scan_qr = 0

        if self.has_found_qr:
            return self.has_found_qr, time.perf_counter() - t0

        for frame in frames:
            blur = cv2.GaussianBlur(frame, (3,3), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            found_codes = self.scan_pyzbar(gray, show_result)
            for qr in found_codes:
                if self.check_qr(qr) and qr not in ls_qr and qr not in name_video_saved:
                    ls_qr.append(qr)
                    # print(qr)
                    self.has_found_qr = self.hasFound(qr)
            
        if len(ls_qr) >= 2 and motion is not None:
            ls_qr.sort(reverse=self.sort_qr_motion(motion))

        if show_result:
            print("QRCode: ", ls_qr)

        return self.has_found_qr, time.perf_counter() - t0