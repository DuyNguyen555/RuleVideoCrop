import numpy as np
import math
import config


def roi_bounds(frame_shape):
        H, W = frame_shape[:2]

        # clamp
        x1r = max(0.0, min(1.0, float(getattr(config, "X1_REL", 0.0))))
        x2r = max(0.0, min(1.0, float(getattr(config, "X2_REL", 1.0))))
        y1r = max(0.0, min(1.0, float(getattr(config, "Y1_REL", 0.0))))
        y2r = max(0.0, min(1.0, float(getattr(config, "Y2_REL", 1.0))))

        # nếu cấu hình ngược >> full frame
        if x2r <= x1r: x1r, x2r = 0.0, 1.0
        if y2r <= y1r: y1r, y2r = 0.0, 1.0

        # start dùng floor, end dùng ceil (end là exclusive)
        x1 = int(math.floor(x1r * W))
        x2 = int(math.ceil (x2r * W))
        y1 = int(math.floor(y1r * H))
        y2 = int(math.ceil (y2r * H))

        # kẹp biên (cho phép end == W/H vì slice end là exclusive)
        x1 = max(0, min(x1, W-1))
        y1 = max(0, min(y1, H-1))
        x2 = max(x1+1, min(x2, W))
        y2 = max(y1+1, min(y2, H))

        roi_x1, roi_y1 = x1, y1
        roi_x2, roi_y2 = x2, y2
        return roi_x1, roi_y1, roi_x2, roi_y2

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


