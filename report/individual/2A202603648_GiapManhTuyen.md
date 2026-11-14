# Báo cáo Day 7 — Giap Manh Tuyen, 2A202603648

## Dữ liệu và phương pháp

Corpus Day 6 gồm hai bài báo cơ học lượng tử: bản arXiv 1204.0653v1 có 28 trang nội dung hợp lệ và bài Entropy 2023, 25, 213 có 13 trang nội dung hợp lệ. Nguồn, hash và thông tin quyền sử dụng nằm trong `data/corpus_manifest.json` của Day 6. Bản arXiv do người dùng cung cấp, chưa xác nhận quyền phân phối lại; bài Entropy là CC BY 4.0. Các đoạn văn và index giữ cục bộ, không tự đưa lên repo.

Sử dụng collection Chroma persistent `day06-c` gồm 275 chunk, vector 768 chiều, cosine; query embedding dùng EmbeddingGemma 2 local cùng model, chuẩn hóa L2 và prompt `SearchQuery` của Day 6. Agent có thể dùng Gemini 3.5 Flash Lite hoặc GPT-OSS 120B qua Groq và đổi provider khi gặp giới hạn; trace ghi provider cho từng câu. Hai bảng điểm agent dev đã chấm dưới đây được chạy bằng Gemini. Chi phí API thực tế cần đối chiếu hóa đơn; số token được lưu trong trace, không tự suy ra chi phí bằng 0.

Benchmark gồm 15 câu dev có đáp án, 2 câu dev thiếu bằng chứng, 5 câu test mới có đáp án và 2 câu test mới thiếu bằng chứng. Dev phát triển từ bộ câu hỏi Day 6; sửa đáp án q03, thay một số câu và bỏ filter có thể gắn với nhãn. Test Day 6 đã từng được xem nên không được dùng làm test độc lập. `starter/prepare_benchmark.py` tạo `data/benchmark_day7.json`. Mỗi gold được đối chiếu có trong chunk cùng nguồn và trang; việc một đoạn khớp chuỗi không tự chứng minh toàn bộ câu trả lời được hỗ trợ.

Để kiểm tra các câu thiếu bằng chứng, đã tìm trên toàn bộ 275 chunk: không có từ `clinical`, `participant`, `kelvin` hoặc `laboratory`; các đoạn có `experiment` và `uncertainty` được đọc thêm và chủ yếu nói về ví dụ lịch sử, quang học hoặc quan hệ bất định lượng tử. Việc không thấy từ khóa tự nó không chứng minh thiếu thông tin; kết luận vẫn giới hạn trong nội dung corpus đã index và cần người nộp đối chiếu tài liệu gốc.

Ba phương pháp: dense cosine; BM25 theo token; hybrid RRF hợp nhất thứ hạng dense và BM25 với hằng số 60. Không dùng gold trong agent hoặc truy xuất. Context giữ thứ tự kết quả, bỏ ID trùng và theo budget ký tự; agent có tối đa hai lần search trong benchmark. Các cấu hình và tiêu chí chọn công bố tại `report/dev_selection_criteria.md`.

## Retrieval dev

Định nghĩa Hit@1 và Evidence Hit@3: cùng source ID, trang và evidence sau chuẩn hóa Unicode và khoảng trắng; mẫu số là 15 câu có đáp án. Context Evidence Rate dùng evidence có trong context cuối. Câu thiếu bằng chứng không vào mẫu số. Thời gian là trung bình một query sau warmup, bao gồm thời gian embedding đối với dense và hybrid; chi phí embedding local không tính phí API. Warmup model khoảng 54,7 giây trong lượt chạy này.

| Cấu hình | Hit@1 | Hit@3 | Context Evidence Rate | Thời gian trung bình/query |
|---|---:|---:|---:|---:|
| Dense, top 3, 800 ký tự | 8/15 | 12/15 | 8/15 | 0,266 s |
| BM25, top 3, 800 ký tự | 4/15 | 10/15 | 4/15 | 0,002 s |
| Hybrid RRF, top 3, 800 ký tự | 6/15 | 11/15 | 6/15 | 0,270 s |
| Hybrid RRF, top 5, 2.400 ký tự | 6/15 | 11/15 | 11/15 | 0,260 s |
| Hybrid RRF, top 8, 4.000 ký tự | 6/15 | 11/15 | 13/15 | 0,260 s |

Hai cấu hình context lớn thử giả thuyết giữ thêm bằng chứng. Chỉ số này là kết quả retrieval trên query gốc; agent có thể viết lại query, nên trace agent và chất lượng trả lời phải được xét riêng. Kết quả từng câu và context ở `starter/results/retrieval/dev_traces.json`, tổng hợp ở `dev_summary.json`; cả hai cần được tạo lại trên máy có quyền đọc corpus Day 6.

## Đánh giá agent, lựa chọn và test

Nhãn dưới đây do trợ lý đối chiếu thủ công với trace và nguồn; người nộp cần kiểm tra lại. Điểm câu trả lời tính 0–2. Trace ghi query model tạo, từng search, kết quả, context, câu trả lời, citation, số token, thời gian và lỗi.

| Cấu hình dev | Dùng tool | Điểm 0 / 1 / 2 | Trung bình / 2 | Citation được hỗ trợ | Từ chối đúng câu thiếu evidence | Từ chối nhầm câu có đáp án |
|---|---:|---:|---:|---:|---:|---:|
| BM25 top 3 | 17/17 | 5 / 2 / 8 | 1,20/2 | 10/10 | 2/2 | 5/15 |
| Hybrid top 8 | 17/17 | 2 / 1 / 12 | 1,67/2 | 22/24 | 2/2 | 2/15 |

BM25 dùng một lượt search cho 5 câu, hai lượt cho 12 câu; hybrid dùng một lượt cho 8 câu và hai lượt cho 9 câu. Tổng token API được báo tương ứng là 13.722 và 39.059. Thời gian thực tế trung bình từng case lần lượt khoảng 6,54 và 14,19 giây; gồm cả thời gian chờ giới hạn API và một số lượt khởi tạo model local khi chạy lại, nên không phải phép đo latency ổn định. Nhãn chi tiết nằm trong `report/reviews/`; tổng hợp nằm trong `report/results/`.

- **q04:** BM25 từ chối câu về mục tiêu dẫn xuất đầu tiên; hybrid tìm được đoạn nêu phương trình Schrödinger phụ thuộc thời gian, Hamilton–Jacobi và tiên đề tích phân đường đi, rồi trả lời đúng có nguồn. Đây là một cải thiện quan sát được trên dev.
- **q13:** BM25 trả lời đúng có citation; hybrid tìm được đoạn chứa đủ `m = ω = ℏ = 1` và model cũng nêu đúng, nhưng không đính kèm citation. Khung agent vì vậy từ chối; đây là một lỗi còn lại và là kết quả giảm so với BM25.
- **q12:** Hybrid tìm hai lượt nhưng không đưa gold vào context cuối và từ chối. **q15:** Hybrid nêu đúng trạng thái cân bằng nhiệt, nhưng còn mở rộng sang các trạng thái mà hai citation không xác nhận cho cùng điều kiện bỏ thế; chấm 1/2 và đánh dấu hai citation không hỗ trợ.

Theo tiêu chí công bố trước, chọn **hybrid top 8 / 4.000 ký tự** và dùng **BM25 top 3 / 800 ký tự** làm baseline cho test. Hybrid có Context Evidence Rate 13/15 so với 4/15 và điểm trả lời đã chấm 1,67/2 so với 1,20/2, nhưng cần nhiều token hơn và có tỷ lệ citation được hỗ trợ thấp hơn trong lượt dev. Bản khóa cấu hình và hash benchmark được lưu tại `report/results/selection.json` trước khi chạy test.

Dev agent cho dense, BM25 và hybrid top 8 dùng Gemini trên toàn bộ 17 câu mỗi cấu hình. Hai cấu hình hybrid top 3 và top 5 mỗi cấu hình dùng Gemini 10 câu và Groq 7 câu vì giới hạn API. Vì vậy, sự khác biệt câu trả lời giữa các cấu hình có thể do model; bảng retrieval trên query gốc dùng cùng embedding/index và không phụ thuộc model agent.

Các lượt agent thực tế trên dev có số trạng thái như sau; `answered` chỉ là kết quả qua kiểm tra ID, không có nghĩa nội dung đã được chấm đúng. Hai hàng BM25 và hybrid top 8 có nhãn nội dung ở bảng trên.

| Cấu hình dev | Answered / abstain | Gemini / Groq | Tổng token API được báo |
|---|---:|---:|---:|
| Dense top 3 | 11 / 6 | 17 / 0 | 16.213 |
| BM25 top 3 | 10 / 7 | 17 / 0 | 13.722 |
| Hybrid top 3 | 10 / 7 | 10 / 7 | 18.543 |
| Hybrid top 5 | 12 / 5 | 10 / 7 | 26.607 |
| Hybrid top 8 | 13 / 4 | 17 / 0 | 39.059 |

Ví dụ dev q04: agent hybrid trả lời mục tiêu đầu tiên là dẫn xuất phương trình Schrödinger phụ thuộc thời gian từ phương trình Hamilton–Jacobi và tiên đề tích phân đường đi, trích `[3131d49d4e00a1f7a53c]` từ `data/data.pdf`, trang vật lý 4. Đoạn nguồn đó nêu rõ quan hệ này. Ví dụ thiếu evidence test tu01 hỏi dụng cụ đo điện trường trong một thí nghiệm; agent tìm nguồn rồi trả lời “Không đủ bằng chứng để trả lời.” Lỗi q13 và t02/t03 được phân tích ở các đoạn lân cận.

### Kết quả test sau khi khóa lựa chọn

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian retrieval/query sau warmup |
|---|---:|---:|---:|---:|
| BM25 top 3 | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) | 0,005 s |
| Hybrid top 8 | 3/5 (60%) | 3/5 (60%) | 4/5 (80%) | 0,356 s |

| Agent test | Dùng tool | Điểm 0 / 1 / 2 | Trung bình / 2 | Citation được hỗ trợ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 top 3 | 7/7 | 2 / 0 / 3 | 1,20/2 | 3/3 | 3/3 | 2/2 | 2/5 |
| Hybrid top 8 | 7/7 | 3 / 0 / 2 | 0,80/2 | 2/2 | 2/2 | 2/2 | 3/5 |

Review context cho năm câu có đáp án, theo thứ tự t01–t05: BM25 **2, 0, 2, 2, 0** (ba mức 2 và hai mức 0); hybrid **2, 2, 2, 2, 0** (bốn mức 2 và một mức 0). Mức 0: không có evidence trả lời; mức 1: có nhưng thiếu điều kiện hoặc nguồn; mức 2: đủ evidence và nguồn. Lý do từng câu và nhãn citation nằm trong `report/reviews/test_bm25.json` và `report/reviews/test_hybrid8.json`.

Test agent BM25 có 6 case Groq và 1 case Gemini; hybrid có 4 case Groq và 3 case Gemini. Với năm câu có đáp án, BM25 dùng Groq cho cả 5; hybrid dùng Groq cho t01–t04 và Gemini cho t05. Số token API được báo, bao gồm usage đã nhận ở các lượt thử thất bại, là 7.877 và 17.315; thời gian trung bình lần lượt 2,40 và 7,81 giây/case, gồm khởi tạo và chờ API. Không có lỗi provider chưa phục hồi; mỗi cấu hình có một lượt thử gặp giới hạn và đã chuyển provider. Không suy ra chi phí tiền từ token; kiểm tra hóa đơn của từng dịch vụ.

Hybrid đưa thêm evidence vào context nhưng **trả lời test kém hơn baseline**. Ở t02, model nêu đúng q là vị trí, p là động lượng nhưng đặt khoảng trắng trong một citation, khiến khung từ chối. Ở t03, model nêu đúng tính unitary nhưng parser nhầm ngoặc công thức LaTeX và ID có khoảng trắng thành citation không hợp lệ, nên khung từ chối. Ở t05 cả hai không có đoạn phân rã entropy trong context. Các lỗi này được ghi nhận sau test; không dùng test để điều chỉnh rồi coi kết quả đã điều chỉnh là đánh giá độc lập. Với chỉ năm câu có đáp án và việc chuyển provider, không thể khái quát mạnh về chất lượng trên dữ liệu khác.

Toàn bộ kết quả trên là nhãn do trợ lý đối chiếu với các đoạn nguồn; người nộp cần kiểm tra lại trước khi nộp. Traces đầy đủ nằm tại `starter/results/agent/` và `starter/results/retrieval/`, bị Git bỏ qua vì có đoạn từ corpus; khi chia sẻ bài cần cấp quyền truy cập phù hợp hoặc chạy lại theo hướng dẫn sau khi xác nhận quyền với nguồn. Không dùng mock hoặc tự điền số liệu; lỗi API được báo riêng.

### OECD EmbeddingGemma retrieval bổ sung trên dev

Sau khi tạo dense index cho tài liệu OECD, chạy `advanced/scripts/evaluate_retrieval.py` trên đúng 18 câu dev đã công bố (15 có đáp án, 3 thiếu evidence), không tạo thêm câu hỏi. Trước khi đo, script kiểm tra gold passage có trong đúng trang PDF của index; retriever kiểm tra đủ 841 ID đúng thứ tự, vector hữu hạn 768 chiều và chuẩn hóa L2. Cả ba phương pháp dùng cùng corpus, query, top-6, context packer 5.200 ký tự; embedding query dùng prompt `SearchQuery` local.

| Phương pháp | Hit@1 (có evidence bất kỳ) | Evidence Hit@3 (có evidence bất kỳ) | Đủ toàn bộ evidence ở top 3 | Context Evidence Rate | Giây/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 9/15 (60,0%) | 13/15 (86,7%) | 12/15 (80,0%) | 15/15 (100%) | 0,057 |
| Dense cosine | 8/15 (53,3%) | 14/15 (93,3%) | 13/15 (86,7%) | 14/15 (93,3%) | 0,344 |
| Hybrid RRF | 8/15 (53,3%) | 15/15 (100%) | 13/15 (86,7%) | 15/15 (100%) | 0,288 |

Evidence Hit@3 là có ít nhất một passage gold trong top 3; cột đầy đủ evidence yêu cầu mọi passage gold, kể cả multi-hop. Context Evidence Rate yêu cầu đủ evidence sau khi đóng gói context. Hybrid có recall top-3 cao nhất; BM25 nhanh hơn và có context coverage bằng hybrid. Agent đạt 18/18 trên dev với BM25 và 17/18 với hybrid, nên khóa BM25 cho test. Bộ test mới 30 câu đạt **27/30 (90%)**, đúng ngưỡng bonus: bảng 6/6, ảnh 4/5, biểu đồ 5/5, multi-hop 2/4, text 10/10 (gồm 6 câu không đủ evidence). Ba lỗi: `t-v01` từ chối câu ảnh có thể trả lời; `t-m01` thiếu citation cho claim 0,7% GDP; `t-m02` bỏ số thuế diesel 1.000 VND/lít. Không có lỗi provider; 84 request, 91.602 token, tổng thời gian 84,5 giây. Mỗi câu và citation được đối chiếu với trang PDF gốc hoặc crop đã kiểm tra. Nhãn từng câu: `advanced/benchmark/test_review.json`; traces và metrics retrieval dev: `advanced/results/retrieval-dev-embedding/`; hướng dẫn chạy: `advanced/README.md`.

## Chạy lại và kiểm thử

Lệnh tạo benchmark, chạy retrieval/agent và chuẩn bị review được ghi trong `starter/README.md`; thay đường dẫn Day 6 và model nếu chạy ở máy khác. Khóa test lưu hash benchmark, chỉ cho chạy baseline và cấu hình đã chọn. API key nằm trong `.env` cục bộ, không đưa vào Git.

Đã chạy `$env:PYTHONUTF8='1'; D:\miniconda3\python.exe -m unittest discover -s tests -q`: **25 test đạt**. Các test kiểm tra hợp đồng code, tool call, chuyển provider và khóa test; chúng không thay thế benchmark với model thật ở trên.
