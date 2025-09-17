import os
import time
from typing import Dict, List, Tuple, Any
from io_utils.file_ops import rotate_queue
from io_utils.print_log import log

class NameVideo:
    """
    Quản lý việc đổi tên các cặp ảnh đã lưu dựa trên mã QR và trạng thái chuyển động.
    """
    # Hằng số 
    FRAME_DEL_IMG_THRESHOLD = 100
    QR_WAIT_TIMEOUT = 60 
    QR_WAIT_TIMEOUT_UP = 60
    QR_WAIT_TIMEOUT_DOWN = 150
    
    # Hằng số cho chuyển động và hậu tố file
    MOTION_DOWN = "Down"
    MOTION_UP = "Up"
    QR_SUFFIX_1 = "1"
    QR_SUFFIX_2 = "2"
    FRAME_ID_TOP = "_top"
    FRAME_ID_FRONT = "_front"
    UNKNOWN_SUFFIX_0 = "_unknown_0"
    UNKNOWN_SUFFIX_1 = "_unknown_1"

    def __init__(self):
        # Cờ trạng thái cho việc đổi tên tuần tự 
        self.rename_check_down: bool = False
        self.rename_check_up: bool = False
        self.is_del_up : bool = False
        
        # Trạng thái cho logic chờ QR sai thứ tự
        self.waiting_on_motion: str | None = None
        self.qr_wait_counter: int = 0

        # Bộ đếm cho việc xóa link ảnh
        self.frame_count_link_img: int = 0
        self.sequence_wait_counter: int = 0 
    
    def _reset_wait_state(self):
        """Reset lại trạng thái chờ QR."""
        self.waiting_on_motion = None
        self.qr_wait_counter = 0

    def _execute_rename_and_rotate(self,
                                   output_img_dir: str,
                                   imgs_save: Dict[int, List[str]],
                                   ls_qr: List[str],
                                   name_video_saved: List[str],
                                   pallet_seq: int,
                                   motion: str
                                   ) -> Tuple[Dict[int, List[str]], List[str], List[str], int, bool]:
        """
        Thực hiện việc đổi tên file, xoay vòng hàng đợi và cập nhật trạng thái.
        """
        qr = ls_qr.pop(0)
        name_video_saved.append(qr)

        if motion == self.MOTION_DOWN:
            frame_suffixes = (self.FRAME_ID_TOP, self.FRAME_ID_FRONT)
        elif motion == self.MOTION_UP:
            frame_suffixes = (self.FRAME_ID_FRONT, self.FRAME_ID_TOP)
        else:
            frame_suffixes = (self.UNKNOWN_SUFFIX_0, self.UNKNOWN_SUFFIX_1)
        
        for idx, old_path in enumerate(imgs_save[0]):
            suffix = frame_suffixes[idx]
            _base, ext = os.path.splitext(old_path)
            new_path = os.path.join(output_img_dir, f"{qr}{suffix}{ext}")
            
            if os.path.exists(new_path):
                os.remove(new_path)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

            old = old_path.split("\\")
            new = new_path.split("\\")
            # log("DEBUG", f"Rename: {old[-1]}-->{new[-1]}")

        imgs_save = rotate_queue(imgs_save)

        if pallet_seq > 0:
            pallet_seq -= 1

        return imgs_save, ls_qr, name_video_saved, pallet_seq, True

    def rename_pair_from_queue(self,
                               output_img_dir: str,
                               imgs_save: Dict[int, List[str]],
                               ls_qr: List[str],
                               name_video_saved: List[str],
                               pallet_seq: int,
                               motion: str
                               ) -> Tuple[Dict[int, List[str]], List[str], List[str], int, bool]:
        """
        Điều phối việc kiểm tra và đổi tên cặp ảnh với logic chờ và timeout.
        """
        
        # 1.2 
        if self.rename_check_up and len(ls_qr) == 0:
            self.sequence_wait_counter += 1
            # print(self.sequence_wait_counter)
            if self.sequence_wait_counter >= self.QR_WAIT_TIMEOUT_UP:
                if len(imgs_save) > 1:
                    imgs_save = rotate_queue(imgs_save)
                else:
                    if len(imgs_save[0]) >= 2:
                        imgs_save[0].clear()
                    elif len(imgs_save[0]) > 0:
                        imgs_save[0].clear()
                        self.is_del_up = True

                    # log(" DEBUG", f"Remove 1: {imgs_save[0]}")

                    # Reset lại trạng thái chuỗi
                    self.sequence_wait_counter = 0
                    self.rename_check_up = False
                    self.rename_check_down = False

                return imgs_save, ls_qr, name_video_saved, pallet_seq, False
            
        if self.is_del_up and len(ls_qr) == 0 and len(imgs_save[0]) > 0:
            imgs_save[0].clear()
            self.is_del_up = False
            return imgs_save, ls_qr, name_video_saved, pallet_seq, False

        
        if self.rename_check_down:
            self.sequence_wait_counter += 1
            # print(self.sequence_wait_counter)
            if self.sequence_wait_counter >= self.QR_WAIT_TIMEOUT_DOWN:
                if len(imgs_save[0]) >= 2:
                    # log(" DEBUG", f"Remove 2: {imgs_save[0]}")
                    if len(imgs_save) > 1:
                        imgs_save = rotate_queue(imgs_save)
                    else:
                        imgs_save[0].clear()
                    # Reset lại trạng thái chuỗi
                    self.sequence_wait_counter = 0
                    self.rename_check_up = False
                    self.rename_check_down = False

                return imgs_save, ls_qr, name_video_saved, pallet_seq, False
        
        # Guard clause: Thoát nếu chưa đủ điều kiện
        if len(imgs_save.get(0, [])) != 2 or not ls_qr:
            return imgs_save, ls_qr, name_video_saved, pallet_seq, False
        

        # 1.1 Nếu đang chờ, tăng bộ đếm và kiểm tra timeout
        if self.waiting_on_motion == motion:
            self.qr_wait_counter += 1
            if self.qr_wait_counter >= self.QR_WAIT_TIMEOUT:
                # ls_qr.pop(0)
                # log(" DEBUG", f"Remove 3: {imgs_save[0]}")
                if len(imgs_save) > 1:
                    imgs_save = rotate_queue(imgs_save)
                else:
                    imgs_save[0].clear()
                self._reset_wait_state()


                # Trả về ngay sau khi xóa để chu kỳ sau xử lý QR mới
                return imgs_save, ls_qr, name_video_saved, pallet_seq, False

        # 2. Nếu đổi chiều di chuyển, reset trạng thái chờ
        elif self.waiting_on_motion and self.waiting_on_motion != motion:
            self._reset_wait_state()

        # Logic xử lý QR
        first_qr = ls_qr[0]
        action_taken = False
        is_qr_type_1 = first_qr.endswith(self.QR_SUFFIX_1)
        is_qr_type_2 = first_qr.endswith(self.QR_SUFFIX_2)

        # Chiều lên: Ưu tiên 1 rồi đến 2
        if motion == self.MOTION_UP: 
            if is_qr_type_1:
                self._reset_wait_state()
                self.rename_check_up = True
                self.rename_check_down = False
                action_taken = True

            elif is_qr_type_2:
                 # Đúng thứ tự (đã thấy 1 trước đó)
                if self.rename_check_up:
                    self._reset_wait_state()
                    self.rename_check_up = False
                    action_taken = True

                # Sai thứ tự -> Bắt đầu chờ
                else: 
                    self.waiting_on_motion = self.MOTION_UP

        # Chiều xuống: Ưu tiên 2 rồi đến 1
        elif motion == self.MOTION_DOWN:
            if is_qr_type_2:
                self._reset_wait_state()
                self.rename_check_down = True
                self.rename_check_up = False
                action_taken = True

            elif is_qr_type_1:
                 # Đúng thứ tự (đã thấy 2 trước đó)
                if self.rename_check_down:
                    self._reset_wait_state()
                    self.rename_check_down = False
                    action_taken = True

                # Sai thứ tự -> Bắt đầu chờ
                else: 
                    self.waiting_on_motion = self.MOTION_DOWN

        # Xử lý các QR không phải loại 1 hoặc 2
        if not is_qr_type_1 and not is_qr_type_2:
            self._reset_wait_state()
            self.rename_check_up = False
            self.rename_check_down = False
            action_taken = True

        if action_taken:
            return self._execute_rename_and_rotate(
                output_img_dir, imgs_save, ls_qr, name_video_saved, pallet_seq, motion
            )
            
        return imgs_save, ls_qr, name_video_saved, pallet_seq, False

    def remove_link_img(self, state: Any):
        """
        Xóa link ảnh của pallet đầu tiên nếu không có QR tương ứng sau một khoảng thời gian.
        """
        if len(state.imgs_save.get(0, [])) != 2:
            self.frame_count_link_img = 0
            return

        self.frame_count_link_img += 1
        
        # print(self.frame_count_link_img)

        if self.frame_count_link_img < self.FRAME_DEL_IMG_THRESHOLD:
            return
        
        # log("DEBUG", f"Remove 4: {state.imgs_save[0]}")

        if len(state.imgs_save) > 1:
            state.imgs_save = rotate_queue(state.imgs_save)
        else:
            state.imgs_save[0].clear()
        
        self.frame_count_link_img = 0
    
    def _rename_if_ready(self, state: Any) -> Tuple[Dict[int, List[str]], List[str], List[str], int, float]:
        """
        Phương thức bao bọc logic đổi tên ảnh, dùng để gọi từ bên ngoài.
        """
        t0 = time.perf_counter()
        if state.motion_current not in (self.MOTION_DOWN, self.MOTION_UP):
            return state.imgs_save, state.ls_qr, state.name_video_saved, state.pallet_seq, time.perf_counter() - t0
        
        imgs, ls_qr, name_video_saved, new_pallet_seq, _changed = self.rename_pair_from_queue(
            output_img_dir=state.OUTPUT_IMG, 
            imgs_save=state.imgs_save, 
            ls_qr=state.ls_qr, 
            name_video_saved=state.name_video_saved, 
            pallet_seq=state.pallet_seq, 
            motion=state.motion_current
        )
        
        state.imgs_save = imgs
        state.ls_qr = ls_qr
        state.name_video_saved = name_video_saved
        state.pallet_seq = new_pallet_seq

        return state.imgs_save, state.ls_qr, state.name_video_saved, state.pallet_seq, time.perf_counter() - t0