from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class State:
    # ROI
    roi_ready: bool = False
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 0
    roi_y2: int = 0

    # Thresholds theo ảnh resized
    check_size: bool = False
    y_threshold_up_above: float = 0.0
    y_threshold_up_below: float = 0.0
    y_threshold_between: float = 0.0
    y_threshold_down_above: float = 0.0
    y_threshold_down_below: float = 0.0

    # QR
    scanned_qr: bool = False
    frame_stop_scan: int = 30
    count_qr_1_2: int = 0
    sort: bool = False
    ls_qr: List[str] = field(default_factory=list)
    name_video_saved: List[str] = field(default_factory=list)

    # Crop logic
    crop_2: int = 0
    is_cooldown: bool = False
    rearm_cooldown_frames: int = 10
    count_crop_img_qr: int = 0
    crop_next: bool = False
    wait_pair_active: bool = False
    wait_base: str = ""
    wait_expected_tail: str = ""
    wait_counter: int = 0

    # Motion
    motion_before: str = ""
    motion_current: str = ""
    departure: bool = False

    # Lưu ảnh chờ đổi tên
    imgs_save: Dict[int, List[str]] = field(default_factory=lambda: {0: []})
    pallet_seq: int = 0
    time_reset_crop_img: int = 50

    # Rename
    is_rename_img: bool = False
    time_rename_img: int = 100

    # Directory
    directory_cooldown_frame = 0

    # Rule Up khi lên cao
    up_no_bar_counter: int = 0
    up_no_bar_captured: bool = False

    # Record 
    # name_record: str = ""
    # name_record_start: str = ""
    # name_record_save: List[str] = field(default_factory=list)
    # len_record: int = 0
    # is_change_name_record: bool = False

    # center_y: float = 0.0
    # trigger_start: bool = False
    # trigger_stop: bool = False
    # threshold_clear_name_save: int = 150

    # frame_waitting_qr: int = 0
    # threshold_waitting_qr: int = 100
