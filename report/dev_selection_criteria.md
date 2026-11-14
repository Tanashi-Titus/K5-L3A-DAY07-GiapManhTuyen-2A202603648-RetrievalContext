# Tiêu chí chọn cấu hình Day 7

Chốt cấu hình trên dev trước khi chạy test Day 7. So sánh các phương pháp trên cùng 15 câu dev có đáp án và cùng collection `day06-c`; hai câu dev thiếu bằng chứng được đánh giá riêng về khả năng từ chối. Không dùng filter suy ra từ nhãn.

Ưu tiên cấu hình có Context Evidence Rate cao nhất; nếu bằng nhau ưu tiên Evidence Hit@3, sau đó xem lỗi trả lời có nguồn và từ chối nhầm trên dev, rồi thời gian và số token. Phải giải thích trường hợp chất lượng trả lời hoặc chi phí khiến không chọn cấu hình có context rate cao nhất. Điểm retrieval dùng quy tắc cùng nguồn, trang và chuỗi evidence sau chuẩn hóa Unicode/whitespace; cách này chưa xác nhận evidence đủ ý nên cần đọc thủ công.

Các cấu hình dự kiến gồm dense top-3 / 800 ký tự, BM25 top-3 / 800, hybrid RRF top-3 / 800, hybrid top-5 / 2400 và hybrid top-8 / 4000. Hai cấu hình top-5 và top-8 thử giả thuyết rằng giữ nhiều đoạn và mở rộng context làm tăng tỷ lệ đưa evidence tới agent. Lưu kết quả cả khi giả thuyết không cải thiện.

Toàn bộ lượt chạy agent trong benchmark này dùng cùng cấu hình GPT qua OpenAI. Không chuyển câu sang model khác. Lỗi API được giữ nguyên trong kết quả và tính là lượt không hoàn thành; không tự động gọi lại để thay thế lỗi đó. Retrieval dùng model embedding và index không đổi.
