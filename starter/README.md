# Starter Day 7

## Kết nối Day 6 đang phát triển

`day6_retrieval.py` đọc collection `day06-c` từ Chroma của Day 6 và hỗ trợ dense, BM25 và hybrid RRF. Dense dùng cùng EmbeddingGemma 2, prompt `SearchQuery` và vector 768 chiều như index Day 6. Benchmark đã cấu hình dùng GPT qua OpenAI tool calling. Đặt `OPENAI_API_KEY` và `OPENAI_MODEL` trong `.env` ở thư mục gốc; `.env.example` chỉ chứa tên biến và model mặc định, không có key. Trace ghi model, các lần thử và số token đã báo.

```powershell
$env:PYTHONUTF8='1'
D:\miniconda3\python.exe starter\run_day7.py "What is the Hamilton-Jacobi equation used for?" --provider openai --method hybrid
```

Có thể đổi `--store`, `--embedding-model`, `--env-file`, `--method`, `--top-k`, `--budget` và `--max-searches`. Đường dẫn mặc định trỏ đến máy đã chạy Day 6; cả hai lệnh benchmark đều nhận `--store` và `--embedding-model`, còn lệnh tạo câu hỏi nhận `--day6-root`. Dùng đúng model và prompt query đã tạo index, kể cả khi chuyển model sang đường dẫn khác. Các phụ thuộc Python được ghi trong `requirements-day7.txt` và dùng Miniconda base hiện có. Review thủ công cần đối chiếu với tài liệu nguồn; không dùng kết quả mock để báo cáo chất lượng.

Tạo benchmark và so sánh dev (kết quả và các đoạn nguồn nằm trong `starter/results/`, không được Git theo dõi):

```powershell
D:\miniconda3\python.exe starter\prepare_benchmark.py
D:\miniconda3\python.exe starter\retrieval_benchmark.py --split dev
D:\miniconda3\python.exe starter\agent_benchmark.py --split dev --provider openai --configs bm25-3-800 dense-3-800 hybrid-3-800 hybrid-5-2400 hybrid-8-4000
```

Agent lưu mỗi câu ngay sau khi chạy và có thể chạy lại lệnh để tiếp tục; lỗi API được lưu riêng. Xem `report/dev_selection_criteria.md` trước khi chọn cấu hình và chạy test.

Sau khi chạy đủ các cấu hình dev và tự chấm baseline cùng cấu hình được chọn, khóa lựa chọn bằng `select_config.py` (điền tên cấu hình, đường dẫn hai file review và lý do). Script yêu cầu các trace dev đầy đủ và lưu hash của benchmark. Sau đó:

```powershell
D:\miniconda3\python.exe starter\retrieval_benchmark.py --split test
D:\miniconda3\python.exe starter\agent_benchmark.py --split test --provider openai --output starter\results\gpt\agent --configs <baseline> <cau-hinh-da-chon>
D:\miniconda3\python.exe starter\prepare_reviews.py --split test --config <cau-hinh> --output starter\results\agent\test-review.json
D:\miniconda3\python.exe starter\summarize_agent.py --split test --config <cau-hinh> --reviews starter\results\agent\test-review.json --output starter\results\agent\test-summary.json
```

Thay các giá trị trong dấu `< >`. Review phải đọc context, câu trả lời và nguồn cho từng câu trước khi điền điểm và nhãn citation; script tổng hợp không tự chấm. Kết quả test chỉ dùng đánh giá cấu hình đã chốt, không dùng để điều chỉnh.

Để xem một câu và toàn bộ context, nguồn được trích dẫn phục vụ review: `D:\miniconda3\python.exe starter\show_review.py --split test --config <cau-hinh> --query <id-cau>`. Lệnh này đọc đáp án chuẩn để phục vụ chấm sau khi chạy; đáp án chuẩn không được gửi cho agent.

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
