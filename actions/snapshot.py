import os
import time
from typing import Dict, List, Tuple, Any
from io_utils.file_ops import save_image, del_image
from io_utils.print_log import log


class Snapshot:
    """
    Quản lý logic chụp ảnh (snapshot) dựa trên chuyển động và vị trí của đối tượng.
    Lớp này theo dõi trạng thái, đếm khung hình và quyết định thời điểm thích hợp để lưu ảnh.
    """
    # (Thresholds) 
    Y_THRESHOLD_UP_ABOVE_RATIO   = 0.1
    Y_THRESHOLD_UP_BELOW_RATIO   = 0.4
    Y_THRESHOLD_BETWEEN_RATIO    = 0.5
    Y_THRESHOLD_DOWN_ABOVE_RATIO = 0.8
    Y_THRESHOLD_DOWN_BELOW_RATIO = 1.0

    # --- Hằng số cho việc đếm Frame ---
    # Số frame chờ trước khi xóa ảnh nếu không có pallet đi lên
    UP_NO_PALLET_TIMEOUT_FRAMES = 100
    # Số frame chờ để chụp ảnh thứ hai khi xe đi xuống lúc khởi hành
    DEPARTURE_DOWN_TIMEOUT_FRAMES = 45
    # Số frame chờ để chụp ảnh sau khi pallet đã qua ngưỡng (hướng xuống)
    SNAP_TIMEOUT_DOWN_FRAMES = 20
    # Số frame chờ để chụp ảnh sau khi pallet đã qua ngưỡng (hướng lên)
    SNAP_TIMEOUT_UP_FRAMES = 28
    # Số frame chờ để chụp ảnh khi pallet ở trên đỉnh (hướng lên)
    PALLET_AT_TOP_TIMEOUT_FRAMES = 100
    # Số frame chờ để chụp ảnh front face pallet khi đi xuống dưới cùng nếu có thanh màu cam
    SNAP_TIMEOUT_DOWN_FRAME_QR1 = 80
    
    # --- Hằng số cho logic lưu ảnh ---
    MAX_IMAGES_PER_PALLET = 2


    def __init__(self):
        # --- Cờ trạng thái (State Flags) ---
        self.is_initialized: bool = False              # Đã khởi tạo ngưỡng hay chưa
        self.is_taking_snapshot: bool = False          # Cờ cho biết đang trong quá trình chờ chụp ảnh
        self.is_pallet_at_top: bool = False            # Cờ cho biết pallet đã ở vị trí trên cùng
        self.departure_snap_taken: bool = False        # Cờ cho biết đã chụp ảnh lúc khởi hành hay chưa
        self.is_departure_up_snap_taken: bool = False  # Cờ cho ảnh khởi hành lúc đi lên
        self.is_waiting_for_up_no_pallet: bool = False # Cờ cho biết đang chờ kiểm tra pallet có đi lên không

        # --- Bộ đếm khung hình (Frame Counters) ---
        self.snapshot_frame_counter: int = 0          # Đếm frame cho việc chụp ảnh sau khi qua ngưỡng
        self.frame_top_up_counter: int = 0            # Đếm frame khi pallet ở đỉnh (hướng lên)
        self.frame_top_down_counter: int = 0          # Đếm frame khi khởi hành (hướng xuống)
        self.frame_up_no_pallet_counter: int = 0      # Đếm frame để kiểm tra pallet không đi lên
        self.frame_count_up_no_pallet: int = 2        # Đếm frame xoá ảnh trong trường hợp đặc biệt
        self.frame_count_down_qr1: int = 0

        # --- Trạng thái chuyển động ---
        self.previous_motion: str = ""                # Lưu chuyển động của frame trước đó



    def _initialize_thresholds(self, resized_h: int, state: Any):
        """Khởi tạo các ngưỡng y"""
        state.y_threshold_up_above   += resized_h * self.Y_THRESHOLD_UP_ABOVE_RATIO
        state.y_threshold_up_below   += resized_h * self.Y_THRESHOLD_UP_BELOW_RATIO
        # state.y_threshold_between    += resized_h * self.Y_THRESHOLD_BETWEEN_RATIO
        # state.y_threshold_down_above += resized_h * self.Y_THRESHOLD_DOWN_ABOVE_RATIO
        # state.y_threshold_down_below += resized_h * self.Y_THRESHOLD_DOWN_BELOW_RATIO
        self.is_initialized = True


    def save_snapshot(self, 
                      origin_frame: Any,
                      frame_index_str: str,
                      output_img_dir: str,
                      imgs_save: Dict[int, List[str]],
                      pallet_seq: int) -> Tuple[int, Dict[int, List[str]], str]:
        """
        Lưu một khung hình và quản lý việc phân loại ảnh vào các ngăn pallet.

        Returns:
            Tuple[int, Dict[int, List[str]], str]: (pallet_seq mới, dict ảnh mới, đường dẫn ảnh đã lưu)
        """
        # Đảm bảo key hiện tại tồn tại trong từ điển
        if pallet_seq not in imgs_save:
            imgs_save[pallet_seq] = []

        # Nếu ngăn hiện tại đã đủ ảnh, tạo ngăn mới
        if len(imgs_save[pallet_seq]) >= self.MAX_IMAGES_PER_PALLET:
            pallet_seq += 1
            if pallet_seq not in imgs_save:
                imgs_save[pallet_seq] = []

        img_path = os.path.join(output_img_dir, f"{frame_index_str}.jpeg")
        save_image(img_path, origin_frame)
        imgs_save[pallet_seq].append(img_path)

        # log("DEBUG", f"Save: {frame_index_str}.jpeg")

        # Trả về các giá trị đã cập nhật đúng như docstring mô tả
        return pallet_seq, imgs_save, img_path


    def _handle_motion_change(self, current_motion: str):
        """Xử lý khi có sự thay đổi về trạng thái chuyển động."""
        if self.previous_motion != current_motion and current_motion in ("Down", "Up"):
            self.previous_motion = current_motion
            self.departure_snap_taken = False


    def _handle_up_no_pallet_timeout(self, state: Any):
        """Xử lý trường hợp không có pallet đi lên sau một khoảng thời gian."""
        if not self.is_waiting_for_up_no_pallet:
            return

        self.frame_up_no_pallet_counter += 1
        if self.frame_up_no_pallet_counter >= self.UP_NO_PALLET_TIMEOUT_FRAMES:
            # Hàm del_image sẽ quyết định có xóa ảnh hay không và trả về trạng thái mới
            self.is_waiting_for_up_no_pallet, self.frame_count_up_no_pallet = del_image(
                state.imgs_save, 
                state.pallet_seq, 
                self.frame_count_up_no_pallet
            )
            
            if not self.is_waiting_for_up_no_pallet:
                self.frame_up_no_pallet_counter = 0


    def _handle_departure_snapshot(self, frame: Any, index_frame: int, state: Any) -> bool:
        """Chụp ảnh khi camera bắt đầu di chuyển (khởi hành)."""
        if self.departure_snap_taken:
            return False

        if state.motion_current == "Down":
            if self.frame_top_down_counter == 0:
                self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_0_0",
                                   state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                self.frame_top_down_counter += 1
                return True
            
            self.frame_top_down_counter += 1
            if self.frame_top_down_counter >= self.DEPARTURE_DOWN_TIMEOUT_FRAMES:
                self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_0_1",
                                   state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                self.departure_snap_taken = True
                self.frame_top_down_counter = 0
                return True

        elif state.motion_current == "Up" and not self.is_departure_up_snap_taken:
            self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_0",
                               state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
            
            self.is_departure_up_snap_taken = True
            self.is_waiting_for_up_no_pallet = True
            return True
        
        return False


    def _handle_running_snapshot(self, frame: Any, index_frame: int, y_bar: float, state: Any) -> bool:
        """Chụp ảnh khi camera đang trong hành trình và không có thanh ngang (y_bar == 0.0)."""
        if y_bar != 0.0 :
            self.frame_top_up_counter = 0 # Reset nếu thấy lại thanh ngang
            # Logic đặc biệt khi đi xuống không hết còn xuất hiện mã qr khi di chuyển sang bên
            if state.motion_current == "Down" and state.ls_qr[0].endswith("1"):
                self.frame_count_down_qr1 += 1
                if self.frame_count_down_qr1 >= self.SNAP_TIMEOUT_DOWN_FRAME_QR1:
                    self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_2_2",
                                       state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                    self.frame_count_down_qr1 = 0
                    return True

        # Logic khi không thấy thanh ngang 
        if state.motion_current == "Down":
            if self.is_taking_snapshot:
                self.snapshot_frame_counter += 1
                if self.snapshot_frame_counter >= self.SNAP_TIMEOUT_DOWN_FRAMES:
                    self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_2",
                                       state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                    
                    self.is_taking_snapshot = False
                    self.snapshot_frame_counter = 0
                    self.frame_count_down_qr1 = 0
                    return True
        
        elif state.motion_current == "Up":
            # print(self.is_taking_snapshot)
            if self.is_taking_snapshot:
                self.snapshot_frame_counter += 1
                # print(self.snapshot_frame_counter)
                if self.snapshot_frame_counter >= self.SNAP_TIMEOUT_UP_FRAMES:
                    self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_2",
                                       state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                    
                    self.is_taking_snapshot = False
                    self.snapshot_frame_counter = 0
                    self.is_pallet_at_top = True
                    return True
            
            elif self.is_pallet_at_top:
                self.frame_top_up_counter += 1
                if self.frame_top_up_counter >= self.PALLET_AT_TOP_TIMEOUT_FRAMES:
                    self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_3",
                                       state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
                    
                    self.is_pallet_at_top = False
                    self.frame_top_up_counter = 0
                    return True
        
        return False


    def _handle_y_bar_threshold_snapshot(self, frame: Any, index_frame: int, y_bar: float, has_white: bool, state: Any) -> bool:
        """Chụp ảnh khi thanh ngang (y_bar) đi vào vùng ngưỡng."""
        
        is_in_threshold = state.y_threshold_up_above < y_bar < state.y_threshold_up_below
        if is_in_threshold and not self.is_taking_snapshot and has_white:
            self.save_snapshot(frame, f"{index_frame}_{state.motion_current}_1",
                               state.OUTPUT_IMG, state.imgs_save, state.pallet_seq)
            self.is_taking_snapshot = True
            # Rule _handle_up_no_pallet_timeout
            if state.motion_current == "Up" and self.is_waiting_for_up_no_pallet and self.frame_count_up_no_pallet == 2:
                self.is_waiting_for_up_no_pallet = False
                self.frame_up_no_pallet_counter = 0

            return True
        
        return False


    def snapshot_video(self, frame: Any, index_frame: int, resized_h: int, y_bar: float, has_white: bool, state: Any) -> float:
        """
        Phương thức chính để xử lý mỗi khung hình và quyết định có chụp ảnh hay không.
        """
        t0 = time.perf_counter()

        # 1. Khởi tạo ngưỡng ở lần chạy đầu tiên
        if not self.is_initialized:
            self._initialize_thresholds(resized_h, state)

        # 2. Xử lý khi đổi hướng di chuyển
        self._handle_motion_change(state.motion_current)

        # 3. Xử lý logic xóa ảnh nếu không có pallet đi lên
        self._handle_up_no_pallet_timeout(state)
        
        # 4. Logic chụp ảnh khởi hành
        if self._handle_departure_snapshot(frame, index_frame, state):
            return time.perf_counter() - t0

        # 5. Logic chụp ảnh khi đang chạy (dựa vào y_bar)
        if y_bar == 0.0:
            if self._handle_running_snapshot(frame, index_frame, y_bar, state):
                return time.perf_counter() - t0
        else:
            # Logic khi y_bar có giá trị
            if self._handle_y_bar_threshold_snapshot(frame, index_frame, y_bar, has_white, state):
                return time.perf_counter() - t0
        
        return time.perf_counter() - t0