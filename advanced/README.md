# OECD Economic Surveys: Viet Nam 2025 — Advanced Day 7

## Source and rights

- OECD (2025), *OECD Economic Surveys: Viet Nam 2025*, June 2025, Volume 2025/16, 139 physical PDF pages.
- Official PDF: <https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/06/oecd-economic-surveys-viet-nam-2025_f2511b78/fb37254b-en.pdf>
- DOI: <https://doi.org/10.1787/fb37254b-en>
- OECD states the publication is under CC BY 4.0; third-party material may be excluded. The downloaded source PDF and derived full-page crops stay local and are ignored by Git. Cite OECD and check each figure's original source before redistributing a crop.

## Index design

`scripts/build_index.py` extracts the embedded text layer from all 139 physical pages and splits it into 715 page-linked chunks. Each record keeps both the one-based PDF page and printed page label. All pages with a figure, table or large embedded visual are rendered into `index/visual_pages/`; 96 pages have a visual record and source-page crop. The table data remains indexed with its source page so the original row and column layout can be checked. `index/visual_notes.jsonl` contains human visual readings for benchmark figures and points to the corresponding page crop. `index/manifest.json` records the source hash, coverage and limitations.

Generate an optional local dense index with `D:\miniconda3\python.exe advanced\scripts\embed_index.py`. It uses the existing Day 6 EmbeddingGemma 2 model with the `Document` prompt, 768-dimensional L2-normalized vectors, and local-only inference. The index includes 715 text chunks, 16 table records, 96 visual-page text records, and 14 human-reviewed visual notes (841 records total); image pixels themselves are not embedded. Vectors and IDs are in `index/embeddinggemma2_vectors.npz`; source/page metadata is in `index/embeddinggemma2_records.jsonl`. Dense queries use the asymmetric `SearchQuery` prompt.

The report is text-native rather than a scan, so the pipeline uses its text layer and does not apply OCR. Chart and photograph page crops were inspected directly. Four pages have fewer than 80 extracted characters: physical pages 1 and 3 are cover/title pages, and pages 2 and 95 have no readable text. Pages 1 and 3 are not treated as evidence-bearing visuals. Five separate `visual` questions cover four non-decorative photographs (urban waterfront, family, electric vehicles, factory); chart questions are a separate group and test numeric/series readings. Human-reviewed image notes retain the physical page and crop path.

## Benchmark

`benchmark/questions.json` contains 18 dev cases (15 answerable, 3 evidence-absence) and 30 test cases (24 answerable, 6 evidence-absence). Test answerable groups meet the numerical minima: 6 table, 5 visual, 5 chart, 4 multi-hop and 4 text. Each gold passage is checked against the cited physical PDF page; figure readings include the page crop and a manual observation note. Multi-hop questions cite each required evidence page or table. The evaluation runner sends only case IDs and questions to the model; expected answers, gold passages and absence reasons are not passed to retrieval or generation.

Generate the full index and benchmark with Miniconda base:

```powershell
D:\miniconda3\python.exe advanced\scripts\build_index.py
D:\miniconda3\python.exe advanced\scripts\prepare_benchmark.py
```

The scored pipeline combines page-level lexical BM25 retrieval with manually inspected photo/chart notes. Dense cosine and hybrid BM25+dense RRF are also implemented and compared on dev. Locked agent-evaluation traces use the OpenAI GPT-5.4 mini function-calling controller.

Run dev first:

```powershell
D:\miniconda3\python.exe advanced\scripts\run_advanced.py --split dev --provider openai
```

Launch the GPT-only interactive demo UI with:

```powershell
D:\miniconda3\python.exe -m streamlit run advanced\demo_app.py
```

Then open <http://localhost:8501>. Choose a sample dev question or enter one, and expand the retrieved evidence to show the source text and page crop.

Review all dev traces against the page crops and lock the chosen settings and benchmark hash in `benchmark/selection.json` before running test. The test runner refuses to run if that file is absent, the benchmark hash changed, or the provider/settings differ from dev. A partial `--limit` run is only a smoke check, never a benchmark score.

## Final GPT evaluation

The selected configuration is GPT-5.4 mini, top-k 6, a 5,200-character evidence budget and up to 3 searches. Dev review scored 18/18 (15/15 answerable, 3/3 unanswerable).

The complete fresh 30-question test scored **27/30 (90.0%)**, meeting the bonus threshold of 27/30. All 6 unanswerable questions abstained correctly; 21/24 answerable questions passed the strict answer-and-citation rubric. Group scores, including the unanswerable questions in text, were: tables 6/6, photos/visual 4/5, charts 5/5, multi-hop 2/4 and text 10/10. The misses were `t-v01` (abstained on an answerable waterfront photo), `t-m01` (the 0.7%-of-GDP claim lacked a supporting citation), and `t-m02` (gave the diesel-tax recommendation but omitted the requested 1,000 VND/litre amount). There were no provider errors. Each response and citation was manually checked against its original PDF page or reviewed crop; see `benchmark/test_review.json`.

The 30 cases used 84 model/tool requests and 91,602 reported tokens (88,747 prompt; 2,855 completion). Total model time was 84.5 seconds (2.82 seconds per case on average). The final trace is in `results/openai-test-bonus-final2/test_traces.json`; this is the sole scored test trace retained for submission. Per-question manual labels, reasons and source evidence are in `benchmark/test_review.json`; the selected run and aggregate counts are in `benchmark/selection.json`. The controller does not expose a reliable aggregate API cost, so no dollar estimate is claimed.

## Embedding and retrieval verification

The local EmbeddingGemma 2 index was checked against the current four searchable JSONL files before retrieval: all **841 IDs match in the same order**, all vectors are finite, 768-dimensional and L2-normalized, and the manifest's input hash matches the index. The document/query prompts are `Document` and `SearchQuery`; no embedding API is called. `DenseOecdRetriever` rejects a stale or misordered index and applies exact metadata filters. BM25 and dense use the same filter semantics; hybrid passes the filter to both before RRF.

To compare the assignment's three retrieval methods on the published dev split (not a newly invented question set), run:

```powershell
D:\miniconda3\python.exe advanced\scripts\evaluate_retrieval.py
```

This uses all 18 published dev questions (15 answerable, 3 evidence-absence). It first verifies every gold passage exists in the indexed source page, then records top-6 retrieval and the standard 5,200-character `pack_context`. It reports Hit@1, Evidence Hit@3, complete multi-evidence coverage at 3, context evidence rate, group breakdown and retrieval time. The full run is in `results/retrieval-dev-embedding/summary.json` and `traces.json`.

| Retrieval method | Hit@1 (any evidence) | Evidence Hit@3 (any evidence) | All required evidence in top 3 | Context Evidence Rate | Mean retrieval time/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 9/15 (60.0%) | 13/15 (86.7%) | 12/15 (80.0%) | 15/15 (100%) | 0.057 s |
| EmbeddingGemma 2 dense cosine | 8/15 (53.3%) | 14/15 (93.3%) | 13/15 (86.7%) | 14/15 (93.3%) | 0.344 s |
| Hybrid BM25 + dense RRF | 8/15 (53.3%) | 15/15 (100%) | 13/15 (86.7%) | 15/15 (100%) | 0.288 s |

Evidence Hit@3 counts a query when at least one exact gold passage appears in the top three; the separate complete-evidence column requires every gold passage, including all multi-hop evidence. Context Evidence Rate requires all gold passages to survive context packing. Mean times include query encoding and the first query after model load; embedding inference is local. Hybrid has the best top-3 evidence recall; BM25 is faster and matches its complete context coverage. The agent's locked dev score was 18/18 for BM25 and 17/18 for hybrid, so **BM25 was selected for the test**. The 27/30 test result is the locked BM25 run; no test questions were used to tune retrieval.
