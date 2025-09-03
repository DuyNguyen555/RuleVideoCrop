import cv2
import time
import numpy as np

# frame_directory_up = 0
# frame_directory_down = 0
# def update_direction(y_bar, directory, prev_y_bar):
#     global frame_directory_up, frame_directory_down
#     if prev_y_bar is not None:
#         if y_bar < prev_y_bar:
#             if frame_directory_down >= 5:
#                 directory = "down"
#                 frame_directory_up = 0
#             else:
#                 frame_directory_down += 1

#         elif y_bar > prev_y_bar:
#             if frame_directory_up >= 5:
#                 directory = "up"
#                 frame_directory_down = 0
#             else:
#                 frame_directory_up += 1
                
#     prev_y_bar = y_bar
#     return directory, prev_y_bar

class MotionDetector:
    def __init__(
        self,
        stability_frames: int = 10,        # số frame liên tiếp để chốt hướng
        move_threshold: float = 0.1,       # độ dịch chuyển (px) tối thiểu để tính là có chuyển động
        retrack_threshold: int = 10,       # nếu số điểm theo dõi < ngưỡng thì retrack
        axis_dominance_ratio: float = 1.1, # trục chi phối: |dx| >= 1.2*|dy| → trái/phải; |dy| >= 1.2*|dx| → lên/xuống
        prefer_vertical: bool = True       # nếu True, ưu tiên phân loại Up/Down khi độ lớn tương đương
    ):
        # Lucas-Kanade OF params
        self.feature_params = dict(maxCorners=150, qualityLevel=0.1, minDistance=5, blockSize=7)
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.move_threshold       = move_threshold
        self.retrack_threshold    = retrack_threshold
        self.axis_dominance_ratio = axis_dominance_ratio
        self.prefer_vertical      = prefer_vertical

        # smoothing/stability
        self.DIRECTION_CONFIRM_FRAMES = stability_frames
        self.candidate_direction = "None"
        self.candidate_count = 0
        self.confirmed_direction = "None"

        # OF state
        self.prev_gray = None
        self.p0 = None

    def reset_tracks(self, gray):
        self.prev_gray = gray
        self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        self.candidate_direction = "None"
        self.candidate_count = 0

    def reset_direction(self):
        """
        Reset toàn bộ trạng thái hướng
        """
        self.candidate_direction = "None"
        self.candidate_count = 0
        self.confirmed_direction = "None"
        self.prev_gray = None
        self.p0 = None

    def _classify_direction(self, dx: float, dy: float) -> str:
        """
        Phân loại dựa trên độ chi phối của trục:
        - Nếu |dx| >= ratio * |dy| -> Left/Right
        - Nếu |dy| >= ratio * |dx| -> Up/Down
        - Nếu không trục nào chi phối rõ -> ưu tiên theo prefer_vertical hoặc None
        Lưu ý: hệ trục ảnh OpenCV: y tăng xuống dưới → dy> 0 là "Down", dy<0 là "Up".
        """
        adx, ady = abs(dx), abs(dy)

        # Ngưỡng chuyển động tối thiểu
        if max(adx, ady) < self.move_threshold:
            return "None"

        # Trục chi phối
        horiz_dominant = adx >= self.axis_dominance_ratio * ady
        vert_dominant  = ady >= self.axis_dominance_ratio * adx

        if horiz_dominant:
            return "Left" if dx > 0 else "Right" 
        if vert_dominant:
            return "Up" if dy > 0 else "Down" 

        # Không rõ ràng – xử lý theo ưu tiên
        if self.prefer_vertical:
            if ady >= self.move_threshold:
                return "Down" if dy > 0 else "Up"
        else:
            if adx >= self.move_threshold:
                return "Right" if dx > 0 else "Left"
        return "None"

    def update(self, frame):
        """Trả về 1 trong: 'Up' | 'Down' | 'Left' | 'Right' | 'None'."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # (Re)track nếu thiếu điểm
        if self.p0 is None or self.prev_gray is None or len(self.p0) < self.retrack_threshold:
            self.reset_tracks(gray)
            if self.p0 is None:
                return self.confirmed_direction
            
            self.prev_gray = gray
            return self.confirmed_direction 

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

        raw_direction = "None"
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            if len(good_new) > 0:
                # Dịch chuyển trung bình
                dx, dy = np.mean(good_new - good_old, axis=0)
                raw_direction = self._classify_direction(float(dx), float(dy))

                # cập nhật điểm cho frame sau
                self.p0 = good_new.reshape(-1, 1, 2)
            else:
                self.p0 = None  # buộc retrack lần tới

        # Ổn định hướng
        if raw_direction == self.candidate_direction:
            self.candidate_count += 1
        else:
            self.candidate_direction = raw_direction
            self.candidate_count = 1

        if (
            self.candidate_direction != "None"
            and self.candidate_count >= self.DIRECTION_CONFIRM_FRAMES
            and self.confirmed_direction != self.candidate_direction
        ):
            self.confirmed_direction = self.candidate_direction
            # print(f"** Change: {self.confirmed_direction} **")

        self.prev_gray = gray
        return self.confirmed_direction
    

cooldown_departure = 0
direction = "None"
def FindMotion(motion_detector, roi, y_bar, state):
    t0 =time.perf_counter()
    global direction, cooldown_departure

    # print(direction)
    

    if not state.departure:
        direction = motion_detector.update(roi)
        # print(direction)
        if direction in ("Up", "Down"):
            state.departure = True
            return direction, state.departure, time.perf_counter() - t0
            # return direction, state.departure, time.perf_counter() - t0
    else:
        if y_bar == 0.0:
            cooldown_departure += 1
            # print(cooldown_departure)
            if cooldown_departure >= 140 and direction in ("Up", "Down"):
                state.departure = False
                direction = "None"
                motion_detector.reset_direction()
                cooldown_departure = 0
                state.name_video_saved = []
        else:
            cooldown_departure = 0

        return direction, state.departure ,time.perf_counter() - t0
        # return direction, state.departure, time.perf_counter() - t0
    
    return direction, state.departure, time.perf_counter() - t0
    # return direction, state.departure, time.perf_counter() - t0

    # print(direction)
    # print(time.perf_counter() - t0)