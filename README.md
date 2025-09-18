# Hệ thống Xử lý Video Tự động

## Tổng quan

Hệ thống này là một ứng dụng xử lý video tự động, được thiết kế để hoạt động liên tục, tiếp nhận các tác vụ xử lý video từ một hàng đợi tin nhắn (`Kafka`). Chức năng cốt lõi của hệ thống là phân tích nội dung của từng video, phát hiện các sự kiện quan trọng dựa trên chuyển động và các dấu hiệu hình ảnh (thanh màu, mã QR), sau đó tự động chụp lại các khung hình chính và đổi tên chúng một cách có ý nghĩa để lưu trữ.

Hệ thống được xây dựng để có thể xử lý đồng thời nhiều video, tối ưu hóa hiệu suất bằng cách sử dụng đa tiến trình.

---

## Kiến trúc và các Thành phần Chức năng

Hệ thống bao gồm các mô-đun chức năng chính sau:

* **Bộ Tiếp nhận Tác vụ (`Worker`)**:
    * Đây là điểm khởi đầu của hệ thống, lắng nghe và nhận các yêu cầu xử lý video từ kênh `Kafka` có tên "videos".
    * Khi nhận được yêu cầu, nó sẽ phân bổ các video cho các tiến trình xử lý riêng biệt để chạy song song, tận dụng tối đa số lõi CPU.

* **Bộ Xử lý Video (`Processor`)**:
    * Chịu trách nhiệm xử lý từng video một cách độc lập.
    * Đọc video theo từng khung hình (frame), có cơ chế bỏ qua frame để tăng tốc độ xử lý.
    * Mỗi khung hình hợp lệ sẽ được đưa qua một **Luồng Xử lý (`Pipeline`)** để phân tích sâu hơn.
    * Ghi lại một tệp nhật ký `.csv` chi tiết về hiệu năng, đo lường thời gian cho từng công đoạn.

* **Luồng Phân tích Khung hình (`Pipeline`)**:
    * Là "bộ não" của việc phân tích hình ảnh, thực hiện một chuỗi các bước trên mỗi khung hình.

    * **Xác định Vùng quan tâm (ROI)**: Tự động cắt ra một vùng cụ thể của khung hình để tập trung phân tích.

    * **Phân tích Màu sắc**: Phát hiện "thanh ngang màu cam" và sau đó tìm các vùng màu trắng bên trong thanh cam đó.
        * [👉 Xem ảnh flow chart -> Phát hiện thanh màu cam](image/orange_bar.png)
        * [👉 Xem ảnh flow chart -> Phát hiện thanh màu trắng](image/detect_white.png)

    * **Phát hiện Mã QR**: Quét và giải mã mã QR từ các vùng màu trắng đã phát hiện.
        * [👉 Xem ảnh flow chart -> Phát hiện mã QR](image/detect_qr.png)

    * **Phát hiện Chuyển động**: Xác định hướng di chuyển chính của camera (ví dụ: `Up`, `Down`, `None`).
        * [👉 Xem ảnh flow chart -> Tìm hướng di chuyển của cam](image/motion_detect.png)

    * **Reset trạng thái khi chuyển động thay đổi**

    * **Rule save frame (`Snapshot`)**:
        * Quyết định thời điểm cần lưu lại một khung hình dựa trên các sự kiện cụ thể.
        * Các sự kiện kích hoạt bao gồm: khi camera bắt đầu di chuyển, khi vật thể đi qua một ngưỡng xác định, hoặc sau một khoảng thời gian chờ sau một sự kiện khác.
        * Các ảnh được lưu tạm thời với tên dựa trên chỉ số khung hình và trạng thái lúc đó.
        * [👉 Xem ảnh flow chart -> Tìm hướng di chuyển của cam](image/snapshot.png)

    * **Rule lưu tên File và xoá tên file trong danh sách (`NameVideo`)**:
        * Quản lý việc đổi tên các ảnh đã chụp một cách thông minh dựa trên các mã QR đã được quét.
        * **Ghép cặp Ảnh và QR**: Hệ thống chờ đến khi có đủ một cặp ảnh và một mã QR tương ứng để thực hiện đổi tên.
        * **Logic thứ tự QR**: Áp dụng các quy tắc về thứ tự xuất hiện của mã QR (ví dụ: đuôi "2" trước, đuôi "1" sau khi đi xuống).
        * **Cơ chế Timeout**: Tự động xóa các ảnh đã lưu nếu không tìm thấy mã QR phù hợp sau một khoảng thời gian chờ, tránh bị "kẹt".
        * **Định dạng tên file cuối cùng**: Đổi tên file theo định dạng `[QR_CODE]_[SUFFIX].jpeg`, với suffix là `_top` hoặc `_front` tùy thuộc vào hướng di chuyển.
        * [👉 Xem ảnh flow chart -> Rule xoá link ảnh](image/remove_link_img.png)
        * [👉 Xem ảnh flow chart -> Rule rename file](image/rename_logic.png)

    * [👉👉 Xem ảnh flow chart -> Pipline](image/pipline.png)

---

## Luồng hoạt động tổng thể

1.  Một tin nhắn chứa đường dẫn đến các file video được gửi vào `Kafka`.
2.  `Worker` nhận tin nhắn và khởi tạo các tiến trình con để xử lý song song các video.
3.  Mỗi tiến trình mở video và đọc từng khung hình.
4.  Mỗi khung hình được đưa qua `Pipeline` để phát hiện chuyển động, thanh màu và mã QR.
5.  Dựa trên các sự kiện, `Snapshot Logic` quyết định lưu lại các khung hình quan trọng vào thư mục tạm.
6.  `NameVideo Logic` liên tục kiểm tra các ảnh đã lưu và danh sách mã QR đã quét.
7.  Khi đủ điều kiện (đủ cặp ảnh, đúng thứ tự QR), nó sẽ đổi tên ảnh theo mã QR tương ứng.
8.  Nếu không đủ điều kiện sau một thời gian chờ, các ảnh không hợp lệ sẽ bị xóa trong hàng đợi.
9.  Quá trình lặp lại cho đến khi xử lý hết video. Một file `log_time.csv` ghi lại hiệu năng xử lý được xuất ra cho mỗi video.

---

## Hướng dẫn chạy

1. Cài đặt Docker Compose bằng lệnh:
```bash
docker-compose up -d
```

2. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

3. Tải các video cần test về rồi sau đó trong file `sendmsg.py` thêm các đường dẫn để khi cần gửi kafka xử lý video:
```bash
videos = {
    "videos": [
        "input/part_0030.mp4",
        "input/part_0120.mp4",
        "input/part_0990.mp4"
    ]
}
```

4. Đảm bảo docker container đã chạy và chạy file `worker.py`:
```bash
python worker.py
```

5. Bắn kafka để xử lý video (tạo 1 terminal khác để chạy):
```bash
python sendmsg.py
```

6. Đợi xử lý và trả về kết quả