import os
from typing import Dict, List, Tuple
from io_utils.file_ops import rotate_queue

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
    if len(imgs_save.get(0, [])) != 2 or len(ls_qr) == 0:
        return imgs_save, ls_qr, name_video_saved, pallet_seq, False

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

    # dồn hàng đợi về phía trước
    imgs_save = rotate_queue(imgs_save)

    # hạ pallet_seq nếu đang > 0 để không trỏ tới key không tồn tại
    if pallet_seq > 0:
        pallet_seq -= 1

    return imgs_save, ls_qr, name_video_saved, pallet_seq, True