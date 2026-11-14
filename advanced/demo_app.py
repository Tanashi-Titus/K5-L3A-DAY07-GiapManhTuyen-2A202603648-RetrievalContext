"""Interactive Streamlit demo for the OECD retrieval agent.

Run from the project root with the Miniconda base Python:
    D:\\miniconda3\\python.exe -m streamlit run advanced\\demo_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "advanced"
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ADV / "scripts"))

from openai_controller import OpenAIController  # noqa: E402
from run_advanced import OecdRetriever  # noqa: E402
from agentic import run_agent  # noqa: E402
from embedding_retriever import (  # noqa: E402
    DenseOecdRetriever,
    HybridOecdRetriever,
    LocalEmbeddingGemma2,
)


class DemoProgressRetriever:
    """Expose each real BM25 lookup as a visible demo stage."""

    def __init__(self, retriever, progress, method_label: str):
        self.retriever = retriever
        self.progress = progress
        self.method_label = method_label
        self.lookup_count = 0

    def search(self, query: str, *, filters=None, top_k: int = 6):
        self.lookup_count += 1
        self.progress.update(
            label=f"Bước 1/3 · {self.method_label} đang tìm evidence (lượt {self.lookup_count})…",
            state="running",
        )
        results = self.retriever.search(query, filters=filters, top_k=top_k)
        self.progress.write(
            f"{self.method_label} đã retrieve **{len(results)} kết quả** cho truy vấn: “{query}”."
        )
        return results


class DemoProgressController:
    """Show when GPT is synthesizing retrieved evidence into its final answer."""

    def __init__(self, controller: OpenAIController, progress):
        self.controller = controller
        self.progress = progress

    def decide(self, question: str, history: list[dict]):
        if history and history[-1].get("context"):
            self.progress.update(
                label="Bước 2/3 · GPT đang đọc và tổng hợp evidence đã retrieve…",
                state="running",
            )
        action = self.controller.decide(question, history)
        if action.get("action") == "search":
            self.progress.write(
                f"GPT yêu cầu retrieve thêm: “{action.get('query', '')}”."
            )
        elif action.get("action") == "answer":
            self.progress.write("GPT đã tổng hợp evidence thành câu trả lời có citation.")
        return action


st.set_page_config(page_title="OECD Viet Nam 2025 · RAG demo", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102b46,#175e68); color: white; margin-bottom: 1.2rem;}
      .hero h1 {font-size: 2rem; margin: 0 0 .45rem 0; color: white;}
      .hero p {margin: 0; opacity: .88;}
      .source-card {border: 1px solid #dce3e8; border-radius: 12px; padding: .9rem 1rem; margin: .6rem 0; background: #fbfcfd;}
      .muted {color: #637381; font-size: .9rem;}
    </style>
    <div class="hero">
      <h1>OECD Economic Surveys: Viet Nam 2025</h1>
      <p>Hỏi báo cáo, kiểm tra các đoạn được truy xuất và mở trang nguồn để đối chiếu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

manifest_path = ADV / "index" / "manifest.json"
if not manifest_path.exists():
    st.error("Chưa có index. Tạo index trước bằng `D:\\miniconda3\\python.exe advanced\\scripts\\build_index.py`.")
    st.stop()

import json

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
with st.sidebar:
    st.subheader("Pipeline")
    retrieval_method = st.selectbox(
        "Phương pháp retrieval",
        [
            "BM25 (baseline đã benchmark)",
            "Dense cosine · EmbeddingGemma 2",
            "Hybrid · BM25 + EmbeddingGemma 2 (RRF)",
        ],
    )
    st.write("BM25 tìm từ khóa; dense tìm tương đồng ngữ nghĩa; hybrid gộp thứ hạng bằng RRF.")
    st.write("**Agent:** OpenAI GPT tool calling")
    st.write(f"**Số trang:** {manifest.get('pdf_pages', '—')}")
    st.write(f"**Text chunks:** {manifest.get('text_chunk_count', '—')}")
    st.write(f"**Trang có visual:** {manifest.get('visual_page_count', '—')}")
    top_k = st.slider("Số kết quả mỗi lượt tìm", 3, 10, 6)
    budget = st.slider("Context budget (ký tự)", 1500, 9000, 5200, step=500)
    max_searches = st.slider("Số lượt tìm tối đa", 1, 5, 3)
    st.caption("Demo dùng dữ liệu index cục bộ và gọi model thật. Không đưa đáp án chuẩn vào agent.")

@st.cache_resource(show_spinner="Nạp index OECD…")
def get_retriever() -> OecdRetriever:
    return OecdRetriever(ADV / "index")


@st.cache_resource(show_spinner="Nạp EmbeddingGemma 2 local…")
def get_dense_retriever() -> DenseOecdRetriever:
    bm25 = get_retriever()
    embedder = LocalEmbeddingGemma2()
    return DenseOecdRetriever(ADV / "index", bm25, embedder)


@st.cache_resource(show_spinner="Chuẩn bị hybrid retriever…")
def get_hybrid_retriever() -> HybridOecdRetriever:
    return HybridOecdRetriever(get_retriever(), get_dense_retriever())


retriever = get_retriever()
benchmark_path = ADV / "benchmark" / "questions.json"
sample_options = {"Tự nhập câu hỏi": ""}
if benchmark_path.exists():
    for case in json.loads(benchmark_path.read_text(encoding="utf-8")):
        if case.get("split") == "dev":
            label = f"{case['id']} · {case['group']} · {case['question']}"
            sample_options[label] = case["question"]
selected_sample = st.selectbox("Câu hỏi mẫu (dev)", list(sample_options))
sample_question = sample_options[selected_sample]
question = st.text_area(
    "Câu hỏi",
    value=sample_question,
    placeholder="Ví dụ: What does Figure 1.2 show about Vietnam’s GDP per capita?",
    key=f"question_{selected_sample}",
    height=95,
)
go = st.button("Tìm trong báo cáo", type="primary", disabled=not question.strip())

if go:
    load_dotenv(ROOT / ".env", override=False)
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL") or "gpt-5.4-mini"
    if not openai_key:
        st.error("Cần OPENAI_API_KEY trong file .env ở thư mục gốc project.")
        st.stop()

    controller = OpenAIController(openai_key, openai_model, max_searches=max_searches, max_retries=1)
    if retrieval_method.startswith("Dense"):
        active_retriever = get_dense_retriever()
        method_label = "EmbeddingGemma 2 dense cosine"
    elif retrieval_method.startswith("Hybrid"):
        active_retriever = get_hybrid_retriever()
        method_label = "Hybrid BM25 + EmbeddingGemma 2 RRF"
    else:
        active_retriever = retriever
        method_label = "BM25"
    progress = st.status("Bước 1/3 · Bắt đầu retrieve evidence…", expanded=True)
    progress.markdown(
        "1. **Retrieve:** GPT chọn truy vấn, BM25 lấy evidence.\n"
        "2. **Tổng hợp:** GPT đọc context đã lấy và quyết định tìm thêm hoặc viết câu trả lời.\n"
        "3. **Kết quả:** hiển thị câu trả lời cuối cùng cùng citation."
    )
    progress_controller = DemoProgressController(controller, progress)
    progress_retriever = DemoProgressRetriever(active_retriever, progress, method_label)
    try:
        result = run_agent(
            question.strip(), progress_controller, progress_retriever,
            max_searches=max_searches, top_k=top_k, budget=budget,
        )
        progress.update(
            label="Hoàn tất · GPT đã tổng hợp câu trả lời từ evidence đã retrieve",
            state="complete",
            expanded=True,
        )
    except Exception as exc:
        progress.update(label="Truy vấn chưa hoàn tất", state="error", expanded=True)
        message = str(exc).replace(openai_key, "[redacted]")
        st.error(f"GPT chưa hoàn tất truy vấn: {message}")
        st.stop()

    searches = [event for event in result.get("events", []) if event.get("event") == "search"]
    if searches:
        st.subheader("Bước 1 · Evidence đã truy xuất")
        for turn, event in enumerate(searches, start=1):
            st.markdown(f"**Lượt {turn}:** `{event['query']}`")
            for rank, hit in enumerate(event.get("results", []), start=1):
                metadata = hit.get("metadata", {})
                physical = metadata.get("pdf_page")
                printed = hit.get("page")
                page_label = f"PDF p. {physical}" if physical else "Trang PDF không rõ"
                if printed:
                    page_label += f" · trang in {printed}"
                st.markdown(
                    f"<div class='source-card'><b>{rank}. {hit.get('id','')}</b> &nbsp;·&nbsp; "
                    f"{page_label} &nbsp;·&nbsp; score {hit.get('score', 0):.3f}</div>",
                    unsafe_allow_html=True,
                )
                left, right = st.columns([3, 2], gap="medium")
                with left:
                    st.write(hit.get("text", ""))
                with right:
                    crop_value = metadata.get("crop_path")
                    if crop_value:
                        crop = (ROOT / crop_value).resolve()
                        visual_root = (ADV / "index" / "visual_pages").resolve()
                        if crop.is_file() and crop.parent == visual_root:
                            st.image(str(crop), caption=f"Nguồn: {page_label}", use_container_width=True)
                        else:
                            st.caption("Có liên kết trang nguồn nhưng ảnh xem trước không có sẵn.")
            st.divider()

    context_ids = list(dict.fromkeys(
        chunk_id for event in searches for chunk_id in event.get("context_ids", [])
    ))
    st.subheader("Bước 2 · GPT tổng hợp evidence")
    synthesis_outcome = (
        "câu trả lời có citation"
        if result["status"] == "answered"
        else "từ chối trả lời do chưa đủ evidence"
    )
    st.write(
        f"GPT đã đọc context gồm **{len(context_ids)} đoạn evidence** qua "
        f"**{len(searches)} lượt tìm**, rồi tạo {synthesis_outcome}."
    )

    st.subheader("Bước 3 · Câu trả lời cuối")
    if result["status"] == "answered":
        st.success("Đã tổng hợp câu trả lời có citation")
    else:
        st.warning("Agent không xác nhận đủ bằng chứng nên từ chối trả lời.")
    st.markdown(result.get("answer") or "Không có câu trả lời.")

    usage = controller.usage
    total_tokens = sum((row.get("usage") or {}).get("totalTokenCount", 0) for row in usage)
    elapsed = sum(row.get("seconds", 0) for row in usage)
    st.caption(f"Provider: OpenAI · model: {controller.model} · {len(searches)} lượt retrieval · {total_tokens} token báo từ API · {elapsed:.1f}s gọi model")
    with st.expander("Trace agent"):
        st.json({"decisions": controller.decisions, "events": result.get("events", [])})

st.markdown(
    "<p class='muted'>Nguồn: OECD (2025), OECD Economic Surveys: Viet Nam 2025. "
    "Đối chiếu câu trả lời với trang gốc trước khi sử dụng.</p>",
    unsafe_allow_html=True,
)
