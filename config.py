# video_path = "video/input/part_0120.mp4"
video_path = "video\input\part_0120.mp4"
output_dir = "video/output" 
x1_rel, x2_rel = 0.2, 0.75
y1_rel, y2_rel = 0.12, 0.6
resize_factor = 12
ls_qr = []
motion_before = ""
motion_current = ""

motion_changeframe = False
name_record = ""
name_record_start = ""
name_record_save = []
len_record = 0
is_change_name_record = False
name_record_save = []
name_video_saved = []

center_y = 0

trigger_start = False
trigger_stop = False

threshold_clear_name_save = 150

frame_waitting_qr = 0
threshold_waitting_qr = 100