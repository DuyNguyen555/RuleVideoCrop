import os
from typing import Dict, List, Tuple
from io_utils.file_ops import save_image  # (đã đổi tên io -> io_utils)

def save_snapshot(origin_frame,
                  frame_index: str,
                  output_img_dir: str,
                  imgs_save: Dict[int, List[str]],
                  pallet_seq: int) -> Tuple[int, Dict[int, List[str]], str]:
    """
    Trả về (pallet_seq_moi, imgs_save_moi, path_ảnh_vừa_lưu)
    - Tự khởi tạo key nếu chưa tồn tại
    - Nếu ngăn hiện tại đã đủ 2 ảnh thì nhảy sang ngăn mới
    """
    # đảm bảo key hiện tại tồn tại
    if pallet_seq not in imgs_save:
        imgs_save[pallet_seq] = []

    # nếu ngăn đã đủ 2 ảnh, tạo ngăn mới
    if len(imgs_save[pallet_seq]) >= 2:
        pallet_seq += 1
        if pallet_seq not in imgs_save:
            imgs_save[pallet_seq] = []

    name_img = os.path.join(output_img_dir, f"{frame_index}.png")
    save_image(name_img, origin_frame)
    imgs_save[pallet_seq].append(name_img)

    return pallet_seq, imgs_save, name_img