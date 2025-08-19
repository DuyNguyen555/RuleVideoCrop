import os

# Input/Output
NAME_VIDEO = "part_0120.mp4"
VIDEO_PATH   = os.path.join("video", "input", NAME_VIDEO)
OUTPUT_DIR   = os.path.join("video", "output")
OUTPUT_IMG   = "img"
# LOG_TIME_CSV = "infor_video"

# ROI - W, H
# X1_REL, X2_REL = 0, 1
# Y1_REL, Y2_REL = 0, 1
X1_REL, X2_REL = 0.2, 0.7
Y1_REL, Y2_REL = 0.1, 0.7

# Resize
RESIZE_FACTOR = 10

# Motion / QR
FRAME_STOP_SCAN = 20
TIME_RESET_CROP_IMG = 20
CROP_2_COOLDOWN_DOWN = 15
CROP_2_COOLDOWN_UP = 20
REARM_COOLDOWN_FRAMES = 10

# Renamer
TIME_RENAME_IMG_DEFAULT = 300

# Màu orange trong HSV
ORANGE_LOWER = (10, 150, 210)
ORANGE_UPPER = (15, 255, 255)

# Directory
DIRECTORY_COOLDOWN = 56 # thời gian để bắt đầu tìm hướng cam lại
WAIT_PAIR_FRAMES = 20   # Rule QRcode có đuôi 1 và 2
UP_NO_BAR_MAX_FRAMES = 30 # Rule khi cam chạy lên trên cùng để lưu ảnh
