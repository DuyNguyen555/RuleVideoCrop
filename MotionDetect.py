import cv2
import time
import numpy as np


class MotionDetector:
    def __init__(self, stability_frames=10, time_without_qr_threshold=3.0):
        # Tham số cho Optical Flow
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.move_threshold = 1.0
        self.retrack_threshold = 40 

        # Số frame liên tiếp để xác nhận hướng mới
        self.DIRECTION_CONFIRM_FRAMES = stability_frames

        # Khoảng thời gian bỏ qua tính toán hướng sau khi vừa thấy QR
        self.TIME_WITHOUT_QR_THRESHOLD = time_without_qr_threshold
        
        self.prev_gray = None
        self.p0 = None
        
        self.displayed_direction = "Stop"
        self.potential_direction = "Stop"
        self.direction_buffer_count = 0
        
        # Biến mới để theo dõi thời gian không thấy QR
        self.last_qr_seen_time = 0

    def update(self, frame, found_qr):
        """
        Cập nhật mỗi frame, quyết định có cần tính toán hay không, và trả về hướng ổn định.
        - Đầu vào:
            - frame: Frame hiện tại.
            - found_qr (bool): True nếu có mã QR được tìm thấy trong frame này. # ko cần thiết
        """
        current_time = time.time()

        if found_qr:
            self.last_qr_seen_time = current_time

        # 2. Quyết định xem có cần chạy optical flow không
        run_optical_flow = True
        # Nếu đang đi lên/xuống VÀ mới thấy QR gần đây -> không cần chạy
        if self.displayed_direction in ["Up", "Down"]:
            if (current_time - self.last_qr_seen_time) < self.TIME_WITHOUT_QR_THRESHOLD:
                run_optical_flow = False

        # Nếu không chạy, chỉ trả về hướng cũ và thoát sớm
        if not run_optical_flow:
            # Vẫn cần cập nhật prev_gray để chuẩn bị cho lần tính toán tiếp theo
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self.displayed_direction

        # 3. Nếu cần thiết, chạy toàn bộ logic optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.p0 is None or self.prev_gray is None or len(self.p0) < self.retrack_threshold:
            self.prev_gray = gray
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.p0 is None: return self.displayed_direction # Không tìm thấy điểm, giữ hướng cũ
            self.direction_buffer_count = 0
            self.potential_direction = "Stop"

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)

        raw_direction = "Stop"
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            if len(good_new) > 0:
                dx, dy = np.mean(good_new - good_old, axis=0)
                if abs(dx) > abs(dy):
                    if dx > self.move_threshold: raw_direction = "Right"
                    elif dx < -self.move_threshold: raw_direction = "Left"
                else:
                    if dy > self.move_threshold: raw_direction = "Up"
                    elif dy < -self.move_threshold: raw_direction = "Down"
            
            self.p0 = good_new.reshape(-1, 1, 2)
        
        if raw_direction == self.potential_direction:
            self.direction_buffer_count += 1
        else:
            self.potential_direction = raw_direction
            self.direction_buffer_count = 1
            
        if self.direction_buffer_count >= self.DIRECTION_CONFIRM_FRAMES:
            if self.displayed_direction != self.potential_direction:
                print(f"** Change: {self.potential_direction} **")
                self.displayed_direction = self.potential_direction
        
        self.prev_gray = gray.copy()
        return self.displayed_direction