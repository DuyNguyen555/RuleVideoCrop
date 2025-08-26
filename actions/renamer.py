import os
from typing import Dict, List, Tuple
from io_utils.file_ops import rotate_queue

rename_check_down = False
rename_check_up = False
rename_action = False

def rename_pair_from_queue(output_img_dir: str,
                           imgs_save: Dict[int, List[str]],
                           ls_qr: List[str],
                           name_video_saved: List[str],
                           pallet_seq: int,
                           motion: str
                           ):
    """
    Trả về (imgs_save_moi, ls_qr_moi, name_video_saved_moi, pallet_seq_moi, changed)
    - Nếu có đủ 2 ảnh ở ngăn 0 và có QR: đổi tên -> rotate queue
    - Sau rotate: nếu pallet_seq > 0 thì giảm 1 (bảo toàn index hiện hành)
    """
    global rename_check_down, rename_check_up, rename_action
    if len(imgs_save.get(0, [])) != 2 or len(ls_qr) == 0:
        return imgs_save, ls_qr, name_video_saved, pallet_seq, False
    
    if motion == "Down":
        rename_check_up = False
        if ls_qr[0].endswith("2"):
            print(1)
            rename_action = True
            rename_check_down = True
        elif ls_qr[0].endswith("1") and rename_check_down:
            print(2)
            rename_action = True
            rename_check_down = False
        elif not ls_qr[0].endswith("2") and not ls_qr[0].endswith("1"):
            print(3)
            rename_action = True

    elif motion == "Up":
        rename_check_down = False
        if ls_qr[0].endswith("1"):
            print(4)
            rename_action = True
            rename_check_up = True
        elif ls_qr[0].endswith("2") and rename_check_up:
            print(5)
            rename_action = True
            rename_check_up = False

        elif not ls_qr[0].endswith("2") and not ls_qr[0].endswith("1"):
            print(6)
            rename_action = True

    if rename_action:
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
            print(old_path, "-->", f"{qr}{suffix}{ext}")
            if os.path.exists(new_path):
                os.remove(new_path)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        # dồn hàng đợi về phía trước
        imgs_save = rotate_queue(imgs_save)

        # hạ pallet_seq nếu đang > 0 để không trỏ tới key không tồn tại
        if pallet_seq > 0:
            pallet_seq -= 1

        rename_action = False
        return imgs_save, ls_qr, name_video_saved, pallet_seq, True
    
    return imgs_save, ls_qr, name_video_saved, pallet_seq, False

