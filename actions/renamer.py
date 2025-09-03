import os
from typing import Dict, List, Tuple
from io_utils.file_ops import rotate_queue


class Rename:
    def __init__(self):
        self.rename_check_down = False
        self.rename_check_up = False
        self.rename_action = False
        
    def rename_pair_from_queue(self,
                            output_img_dir: str,
                            imgs_save: Dict[int, List[str]],
                            ls_qr: List[str],
                            name_video_saved: List[str],
                            pallet_seq: int,
                            motion: str
                            ):
        if len(imgs_save.get(0, [])) != 2 or len(ls_qr) == 0:
            return imgs_save, ls_qr, name_video_saved, pallet_seq, False
        
        if motion == "Down":
            self.rename_check_up = False
            if ls_qr[0].endswith("2"):
                # print(1)
                self.rename_action = True
                self.rename_check_down = True
            elif ls_qr[0].endswith("1") and self.rename_check_down:
                # print(2)
                self.rename_action = True
                self.rename_check_down = False
            elif not ls_qr[0].endswith("2") and not ls_qr[0].endswith("1"):
                # print(3)
                self.rename_action = True

        elif motion == "Up":
            self.rename_check_down = False
            if ls_qr[0].endswith("1"):
                # print(4)
                self.rename_action = True
                self.rename_check_up = True
            elif ls_qr[0].endswith("2") and self.rename_check_up:
                # print(5)
                self.rename_action = True
                self.rename_check_up = False

            elif not ls_qr[0].endswith("2") and not ls_qr[0].endswith("1"):
                # print(6)
                self.rename_action = True

        if self.rename_action:
            qr = ls_qr.pop(0)
            name_video_saved.append(qr)

            if motion == "Down":
                frame_id_0 = "_top"
                frame_id_1 = "_front"
            elif motion == "Up":
                frame_id_0 = "_front"
                frame_id_1 = "_top"
            else:
                frame_id_0 = "_unknown_0"
                frame_id_1 = "_unknown_1"
            
            # print(imgs_save[0])

            for idx, old_path in enumerate(imgs_save[0]):
                suffix = frame_id_0 if idx == 0 else frame_id_1
                base, ext = os.path.splitext(old_path)
                new_path = os.path.join(output_img_dir, f"{qr}{suffix}{ext}")
                # print(old_path, "-->", f"{qr}{suffix}{ext}")
                if os.path.exists(new_path):
                    os.remove(new_path)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)

            imgs_save = rotate_queue(imgs_save)

            # hạ pallet_seq nếu đang > 0 để không trỏ tới key không tồn tại
            if pallet_seq > 0:
                pallet_seq -= 1

            self.rename_action = False
            return imgs_save, ls_qr, name_video_saved, pallet_seq, True
        
        return imgs_save, ls_qr, name_video_saved, pallet_seq, False

