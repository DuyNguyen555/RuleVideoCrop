import numpy as np
import config
import re

def resize_points_to_roi(points: np.ndarray, roi_shape, roi_resized_wh=None, normalized=False):
    """
    points: toạ độ trong ảnh ROI đã resize (hoặc normalized nếu normalized=True)
    roi_shape: (H, W) của ROI gốc (chưa resize)
    """
    rh0, rw0 = roi_shape[:2]
    pts = points.astype(np.float32).copy()

    if roi_resized_wh is not None:
        rw, rh = roi_resized_wh
        if normalized:
            pts[:, 0] = pts[:, 0] * rw0
            pts[:, 1] = pts[:, 1] * rh0
        else:
            sx = rw0 / rw
            sy = rh0 / rh
            pts[:, 0] = pts[:, 0] * sx
            pts[:, 1] = pts[:, 1] * sy
    else:
        if normalized:
            pts[:, 0] = pts[:, 0] * rw0
            pts[:, 1] = pts[:, 1] * rh0
            
    return pts


_QR_PATTERN = re.compile(r"^[A-Z]{2}-\d{3}-\d$")

def check_qr(qr):
    """
    Kiểm tra mã QR: [2 chữ]-[3 số]-[1 số]
    """
    return bool(_QR_PATTERN.match(qr))