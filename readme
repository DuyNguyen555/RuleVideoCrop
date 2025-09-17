# Hệ thống Xử lý Video Tự động

## Tổng quan

[cite_start]Hệ thống này là một ứng dụng xử lý video tự động, được thiết kế để hoạt động liên tục, tiếp nhận các tác vụ xử lý video từ một hàng đợi tin nhắn (`Kafka`)[cite: 3, 5]. [cite_start]Chức năng cốt lõi của hệ thống là phân tích nội dung của từng video, phát hiện các sự kiện quan trọng dựa trên chuyển động và các dấu hiệu hình ảnh (thanh màu, mã QR), sau đó tự động chụp lại các khung hình chính và đổi tên chúng một cách có ý nghĩa để lưu trữ[cite: 21, 23, 31, 72].

[cite_start]Hệ thống được xây dựng để có thể xử lý đồng thời nhiều video, tối ưu hóa hiệu suất bằng cách sử dụng đa tiến trình[cite: 3].

---

## Kiến trúc và các Thành phần Chức năng

Hệ thống bao gồm các mô-đun chức năng chính sau:

* **Bộ Tiếp nhận Tác vụ (`Worker`)**:
    * [cite_start]Đây là điểm khởi đầu của hệ thống, lắng nghe và nhận các yêu cầu xử lý video từ kênh `Kafka` có tên "videos"[cite: 5].
    * [cite_start]Khi nhận được yêu cầu, nó sẽ phân bổ các video cho các tiến trình xử lý riêng biệt để chạy song song, tận dụng tối đa số lõi CPU[cite: 3].

* **Bộ Xử lý Video (`Processor`)**:
    * [cite_start]Chịu trách nhiệm xử lý từng video một cách độc lập[cite: 12].
    * [cite_start]Đọc video theo từng khung hình (frame), có cơ chế bỏ qua frame để tăng tốc độ xử lý[cite: 14, 15].
    * [cite_start]Mỗi khung hình hợp lệ sẽ được đưa qua một **Luồng Xử lý (`Pipeline`)** để phân tích sâu hơn[cite: 16].
    * [cite_start]Ghi lại một tệp nhật ký `.csv` chi tiết về hiệu năng, đo lường thời gian cho từng công đoạn[cite: 13, 14, 17, 18].

* **Luồng Phân tích Khung hình (`Pipeline`)**:
    * [cite_start]Là "bộ não" của việc phân tích hình ảnh, thực hiện một chuỗi các bước trên mỗi khung hình[cite: 23].
    * [cite_start]**Xác định Vùng quan tâm (ROI)**: Tự động cắt ra một vùng cụ thể của khung hình để tập trung phân tích[cite: 23].
    * [cite_start]**Phân tích Màu sắc**: Phát hiện "thanh ngang màu cam" và sau đó tìm các vùng màu trắng bên trong thanh cam đó[cite: 25, 26].
    * [cite_start]**Phát hiện Mã QR**: Quét và giải mã mã QR từ các vùng màu trắng đã phát hiện[cite: 27].
    * [cite_start]**Phát hiện Chuyển động**: Xác định hướng di chuyển chính của camera (ví dụ: `Up`, `Down`, `None`)[cite: 28, 29].

* **Mô-đun Chụp ảnh Thông minh (`Snapshot`)**:
    * [cite_start]Quyết định thời điểm cần lưu lại một khung hình dựa trên các sự kiện cụ thể[cite: 76, 109].
    * [cite_start]Các sự kiện kích hoạt bao gồm: khi camera bắt đầu di chuyển [cite: 91][cite_start], khi vật thể đi qua một ngưỡng xác định [cite: 106][cite_start], hoặc sau một khoảng thời gian chờ sau một sự kiện khác[cite: 78, 79, 98, 101].
    * [cite_start]Các ảnh được lưu tạm thời với tên dựa trên chỉ số khung hình và trạng thái lúc đó[cite: 85, 92, 99].

* **Mô-đun Đổi tên File (`NameVideo`)**:
    * [cite_start]Quản lý việc đổi tên các ảnh đã chụp một cách thông minh dựa trên các mã QR đã được quét[cite: 38].
    * [cite_start]**Ghép cặp Ảnh và QR**: Hệ thống chờ đến khi có đủ một cặp ảnh và một mã QR tương ứng để thực hiện đổi tên[cite: 58, 68].
    * [cite_start]**Logic thứ tự QR**: Áp dụng các quy tắc về thứ tự xuất hiện của mã QR (ví dụ: đuôi "2" trước, đuôi "1" sau khi đi xuống)[cite: 62, 65, 66].
    * [cite_start]**Cơ chế Timeout**: Tự động xóa các ảnh đã lưu nếu không tìm thấy mã QR phù hợp sau một khoảng thời gian chờ, tránh bị "kẹt"[cite: 50, 55, 70, 71].
    * [cite_start]**Định dạng tên file cuối cùng**: Đổi tên file theo định dạng `[QR_CODE]_[SUFFIX].jpeg`, với suffix là `_top` hoặc `_front` tùy thuộc vào hướng di chuyển[cite: 44, 45].

---

## Luồng hoạt động tổng thể

1.  Một tin nhắn chứa đường dẫn đến các file video được gửi vào `Kafka`.
2.  [cite_start]`Worker` nhận tin nhắn và khởi tạo các tiến trình con để xử lý song song các video[cite: 1, 3].
3.  [cite_start]Mỗi tiến trình mở video và đọc từng khung hình[cite: 12, 15].
4.  [cite_start]Mỗi khung hình được đưa qua `Pipeline` để phát hiện chuyển động, thanh màu và mã QR[cite: 16, 23].
5.  [cite_start]Dựa trên các sự kiện, `Snapshot Logic` quyết định lưu lại các khung hình quan trọng vào thư mục tạm[cite: 31, 109].
6.  [cite_start]`NameVideo Logic` liên tục kiểm tra các ảnh đã lưu và danh sách mã QR đã quét[cite: 72, 73].
7.  [cite_start]Khi đủ điều kiện (đủ cặp ảnh, đúng thứ tự QR), nó sẽ đổi tên ảnh theo mã QR tương ứng[cite: 43, 49, 68].
8.  [cite_start]Nếu không đủ điều kiện sau một thời gian chờ, các ảnh không hợp lệ sẽ bị xóa[cite: 50, 51, 55, 71].
9.  Quá trình lặp lại cho đến khi xử lý hết video. [cite_start]Một file `log_time.csv` ghi lại hiệu năng xử lý được xuất ra cho mỗi video[cite: 13, 14].

---