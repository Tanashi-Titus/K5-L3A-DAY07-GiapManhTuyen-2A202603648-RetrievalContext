# Báo cáo nộp bài Day 7 — Giap Manh Tuyen, 2A202603648

## Tóm tắt

Bài gồm phần core tối đa 85 điểm và thử thách nâng cao tối đa 15 điểm. Core dùng collection Day 6; bonus dùng báo cáo OECD dài 139 trang. Cả hai phần có pipeline agent, các phép thử retrieval, benchmark dev/test và review lỗi. Kết quả benchmark là bằng chứng kỹ thuật, không tự quy đổi thành điểm chính thức.

| Phần | Retrieval đã chọn | Kết quả test đã review | Ghi chú |
|---|---|---:|---|
| Core, tối đa 85 điểm | Hybrid RRF, top-8, context 4.000 ký tự | 4/7 câu đạt nghiêm ngặt (57,1%); 2/2 câu không đủ evidence abstain đúng | Benchmark nhỏ: 5 câu có đáp án, 2 không đủ evidence. BM25 baseline đạt 5/7. |
| Bonus, tối đa 15 điểm | BM25 + ghi chú hình đã kiểm tra thủ công | **27/30 (90%)** | Đạt ngưỡng 90%; 21/24 answerable và 6/6 unanswerable đúng. |

## Core benchmark — tối đa 85 điểm

Agent dùng collection `day06-c` (275 chunks, vector 768 chiều) và ba phương pháp dense, BM25, hybrid RRF. Dev gồm 15 câu có đáp án và 2 câu thiếu evidence; test gồm 5 câu có đáp án và 2 câu thiếu evidence. Dev chọn hybrid top-8 theo context evidence 13/15 so với BM25 4/15 và điểm grounded trung bình 1,67/2 so với 1,20/2.

| Cấu hình | Retrieval Hit@3, dev | Context evidence, dev | Grounded answer, dev | Đạt test nghiêm ngặt |
|---|---:|---:|---:|---:|
| BM25 top-3 baseline | 10/15 | 4/15 | 1,20/2 | 5/7 |
| **Hybrid RRF top-8 được chọn** | 11/15 | 13/15 | 1,67/2 | 4/7 |

Core test có 7 câu và model/provider bị trộn do giới hạn API: Gemini 3.5 Flash Lite và GPT-OSS 120B qua Groq. Vì vậy các so sánh agent có yếu tố nhiễu provider; retrieval trên cùng query/index không bị ảnh hưởng bởi model trả lời. Trên test, BM25 trả lời đầy đủ 3/5 câu có đáp án; hybrid trả lời đầy đủ 2/5; cả hai abstain đúng 2/2 câu thiếu evidence. Traces Day 6 chứa đoạn trích từ corpus có quyền phân phối lại chưa được xác nhận nên không đưa vào repo; kết quả tổng hợp và review từng câu được lưu tại `report/results/` và `report/reviews/`. Cần có quyền truy cập corpus Day 6 để tái tạo trace thô.

## Thử thách nâng cao — tối đa 15 điểm

Nguồn là OECD (2025), *OECD Economic Surveys: Viet Nam 2025*, 139 trang. Nguồn, DOI, giấy phép CC BY 4.0, cấu hình và phần chưa xử lý được ghi trong [`advanced/README.md`](../../advanced/README.md). Index gồm 715 text chunks, 16 bảng, 96 mô tả trang trực quan và 14 ghi chú hình người viết kiểm tra thủ công. Dense/hybrid dùng EmbeddingGemma 2 local, 768 chiều; ảnh pixel không được embedding. PDF và page crops không được commit; người chấm có thể tải tài liệu gốc theo link trong README và kiểm chứng bằng số trang/crop path.

Dev có 15 câu có đáp án và 3 câu thiếu evidence. So sánh retrieval-only dùng cùng corpus, top-6 và context 5.200 ký tự:

| Phương pháp | Hit@1 | Evidence Hit@3 | Đủ evidence top-3 | Đủ evidence trong context | Thời gian/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 9/15 | 13/15 | 12/15 | 15/15 | 0,057 s |
| EmbeddingGemma 2 dense cosine | 8/15 | 14/15 | 13/15 | 14/15 | 0,344 s |
| Hybrid BM25 + dense RRF | 8/15 | 15/15 | 13/15 | 15/15 | 0,288 s |

Agent đạt 18/18 trên dev với BM25, so với 17/18 với hybrid; BM25 được khóa trước khi chạy test. Test mới có 30 câu (24 answerable, 6 unanswerable) và đạt **27/30 = 90%**. Theo nhóm: bảng 6/6, ảnh 4/5, biểu đồ 5/5, multi-hop 2/4, text 10/10 (gồm 6 câu unanswerable). Ba lỗi: abstain nhầm câu ảnh `t-v01`; citation không hỗ trợ claim 0,7% GDP ở `t-m01`; thiếu số thuế 1.000 VND/lít ở `t-m02`. Không có provider error. Thời gian tổng 84,5 giây; 91.602 token được báo. Controller không cung cấp tổng chi phí API đáng tin cậy nên không ước tính tiền.

Review từng câu và trace cuối ở `advanced/benchmark/test_review.json`, `advanced/benchmark/selection.json` và `advanced/results/`. Phần bonus dùng GPT-5.4 mini để trả lời; EmbeddingGemma 2 chỉ được dùng cho so sánh dense/hybrid retrieval dev, còn pipeline test khóa là BM25.

## Tái lập và quyền truy cập

- Cài phụ thuộc theo `starter/requirements-day7.txt` và `advanced/requirements.txt`.
- Sao chép `.env.example` thành `.env`, tự điền API key; không commit `.env`.
- Hướng dẫn chạy core: `starter/README.md`; hướng dẫn demo và bonus: `advanced/README.md` và `advanced/DEMO.md`.
- Benchmark, review và mã nguồn đã commit. PDF OECD và crop ảnh được tạo lại từ URL nguồn; dữ liệu Day 6 cần được cấp quyền riêng nếu người chấm cần chạy lại core từ đầu.
