import cv2
import numpy as np

def nothing(x):
    pass

# Đọc ảnh
img = cv2.imread("test2.png")  # thay bằng đường dẫn ảnh của bạn
img = cv2.resize(img, (640, 480))   # resize cho dễ quan sát

# Chuyển sang HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Giá trị mặc định
ORANGE_LOWER = (10, 150, 250)
ORANGE_UPPER = (15, 255, 255)

# Tạo cửa sổ
cv2.namedWindow("Trackbars")

# Tạo trackbars (gán giá trị mặc định luôn)
cv2.createTrackbar("H_min", "Trackbars", ORANGE_LOWER[0], 179, nothing)
cv2.createTrackbar("H_max", "Trackbars", ORANGE_UPPER[0], 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", ORANGE_LOWER[1], 255, nothing)
cv2.createTrackbar("S_max", "Trackbars", ORANGE_UPPER[1], 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", ORANGE_LOWER[2], 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", ORANGE_UPPER[2], 255, nothing)

while True:
    # Lấy giá trị từ trackbars
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    # Tạo mask theo khoảng màu
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Áp mask lên ảnh gốc
    result = cv2.bitwise_and(img, img, mask=mask)

    # Hiển thị
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
