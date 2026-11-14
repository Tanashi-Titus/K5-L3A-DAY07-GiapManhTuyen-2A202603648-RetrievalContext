# Local chatbot demo

Run from the project root with the existing Miniconda base environment. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` locally; never commit `.env`:

```powershell
D:\miniconda3\python.exe -m streamlit run advanced\demo_app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. The app uses the OECD index in `advanced/index/` and OpenAI GPT-5.4 mini with `OPENAI_API_KEY` from the root `.env`. The retrieval selector offers BM25, EmbeddingGemma 2 dense cosine, and hybrid BM25+dense RRF. The locked benchmark pipeline is BM25 and scored 27/30 (90%); dense and hybrid were compared on dev. The app displays retrieved evidence, page references/crops, the GPT synthesis stage, the final answer, and the agent trace. It does not send benchmark answers to the model. Keep the server bound to `127.0.0.1` when demonstrating with local API keys.

Generate local EmbeddingGemma 2 vectors before selecting dense or hybrid retrieval:

```powershell
D:\miniconda3\python.exe advanced\scripts\embed_index.py
```

This writes normalized 768-dimensional vectors for the text chunks, table records, visual-page captions and human-reviewed visual notes. It uses the existing local model path from Day 6 and makes no embedding API calls.

Use a question from `advanced/benchmark/questions.json` only for a non-scored demo. Benchmark evaluation must use the locked runner and preserve the dev/test boundary.
