# Báo cáo nộp bài Day 7 — Giap Manh Tuyen, 2A202603648

## Tóm tắt

Bài gồm phần core tối đa 85 điểm và thử thách nâng cao tối đa 15 điểm. Core dùng collection Day 6; bonus dùng báo cáo OECD dài 139 trang. Các kết quả dưới đây là benchmark kỹ thuật, không tự quy đổi thành điểm chính thức.

| Phần | Cấu hình được chọn | Dev | Test đã review |
|---|---|---:|---:|
| Core, tối đa 85 điểm | Hybrid RRF, top-8, context 4.000 ký tự; GPT-5.4 mini | 28/30 điểm answerable; 2/2 abstain đúng | BM25 baseline 7/7; hybrid 6/7 do một lượt timeout |
| Bonus, tối đa 15 điểm | BM25; GPT-5.4 mini | 18/18 | **27/30 (90%)**, ngưỡng benchmark đạt |

## Core benchmark — tối đa 85 điểm

Agent dùng collection `day06-c` (275 chunks, vector 768 chiều) với retrieval dense, BM25 và hybrid RRF. Cả năm cấu hình dev, cũng như hai cấu hình test, được chạy bằng GPT-5.4 mini. Dev gồm 15 câu có đáp án và 2 câu thiếu evidence; test gồm 5 câu có đáp án và 2 câu thiếu evidence.

| Cấu hình | Retrieval Hit@3, dev | Context evidence, dev | Điểm câu trả lời đã review, dev | Test |
|---|---:|---:|---:|---:|
| BM25 top-3 baseline | 10/15 | 4/15 | 19/30 (1,27/2) | 7/7 |
| **Hybrid RRF top-8 được chọn** | 11/15 | 13/15 | **28/30 (1,87/2)** | 6/7 |

Hybrid được chọn theo tiêu chí dev đã định trước: context evidence đạt 13/15 so với 4/15 của BM25 và điểm câu trả lời đã review cao hơn. Review phát hiện hai citation ở q10 và q12 không hỗ trợ đầy đủ claim tương ứng; q15 trả lời sai. Đây là điểm cần phân biệt với điểm nội dung câu trả lời trong bảng. Hai câu dev không đủ evidence đều được từ chối đúng ở cả hai cấu hình.

Ở test, BM25 trả lời đúng và có nguồn hỗ trợ cho cả 5 câu answerable, đồng thời abstain đúng cả 2 câu thiếu evidence. Hybrid cũng trả lời đúng 5 câu answerable và abstain đúng tu02; lượt tu01 gặp timeout từ API nên được tính là lỗi, không chạy lại. Do test chỉ có 7 câu, kết quả này không đại diện cho độ ổn định trên tập lớn. Traces thô Day 6 chứa đoạn trích từ corpus có quyền phân phối lại chưa được xác nhận nên giữ local; kết quả tổng hợp và review từng câu được lưu trong `report/results/` và `report/reviews/`.

## Thử thách nâng cao — tối đa 15 điểm

Nguồn là OECD (2025), *OECD Economic Surveys: Viet Nam 2025*, 139 trang. Nguồn, DOI, giấy phép CC BY 4.0, cấu hình và phần chưa xử lý được ghi trong [`advanced/README.md`](../../advanced/README.md). Index gồm 715 text chunks, 16 bảng, 96 mô tả trang trực quan và 14 ghi chú hình được kiểm tra thủ công. Dense/hybrid dùng embedding local 768 chiều; ảnh pixel không được embedding. PDF và page crops không được commit; người chấm có thể tải tài liệu nguồn theo link trong README và kiểm chứng bằng số trang/crop path.

Dev có 15 câu có đáp án và 3 câu thiếu evidence. So sánh retrieval-only dùng cùng corpus, top-6 và context 5.200 ký tự:

| Phương pháp | Hit@1 | Evidence Hit@3 | Đủ evidence top-3 | Đủ evidence trong context | Thời gian/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 9/15 | 13/15 | 12/15 | 15/15 | 0,057 s |
| Dense cosine | 8/15 | 14/15 | 13/15 | 14/15 | 0,344 s |
| Hybrid BM25 + dense RRF | 8/15 | 15/15 | 13/15 | 15/15 | 0,288 s |

Agent đạt 18/18 trên dev với BM25, so với 17/18 với hybrid; BM25 được khóa trước khi chạy test. Test có 30 câu (24 answerable, 6 unanswerable) và đạt **27/30 = 90%**. Theo nhóm: bảng 6/6, ảnh 4/5, biểu đồ 5/5, multi-hop 2/4, text 10/10 (gồm 6 câu unanswerable). Ba lỗi: abstain nhầm câu ảnh `t-v01`; citation không hỗ trợ claim 0,7% GDP ở `t-m01`; thiếu số thuế 1.000 VND/lít ở `t-m02`. Không có lỗi provider. Thời gian tổng 84,5 giây; controller báo 91.602 token. Không ước tính chi phí từ số token.

Review từng câu và trace cuối ở `advanced/benchmark/test_review.json`, `advanced/benchmark/selection.json` và `advanced/results/`. Pipeline bonus dùng GPT-5.4 mini; so sánh dense/hybrid retrieval chỉ là kết quả dev, còn cấu hình pipeline test đã khóa là BM25.

## Tái lập và quyền truy cập

- Cài phụ thuộc theo `starter/requirements-day7.txt` và `advanced/requirements.txt`.
- Sao chép `.env.example` thành `.env`, tự điền `OPENAI_API_KEY`; `.env` không được commit.
- Hướng dẫn core: `starter/README.md`; hướng dẫn demo và bonus: `advanced/README.md` và `advanced/DEMO.md`.
- Kết quả benchmark, review và mã nguồn được commit. Tài liệu OECD và crop ảnh có thể dựng lại từ URL nguồn; dữ liệu Day 6 cần được cấp quyền riêng nếu người chấm cần tái tạo core từ đầu.
