"""Run the advanced OECD benchmark with the real Groq tool-calling model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ADV = ROOT / "advanced"
sys.path.insert(0, str(ROOT / "starter"))
from agentic import run_agent
from gemini_controller import GeminiController
from groq_controller import GroqController
from openai_controller import OpenAIController

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'’][a-z0-9]+)?", re.I)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    def normalize(token: str) -> str:
        token = token.casefold()
        # Small, deterministic English morphology normalization for BM25. This
        # makes queries such as "surpassed" match source wording "surpassing".
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                token = token[:-len(suffix)]
                if len(token) > 3 and token[-1] == token[-2]:
                    token = token[:-1]
                return token
        return token
    return [normalize(x) for x in TOKEN_RE.findall(text)]


class OecdRetriever:
    def __init__(self, index_dir: Path):
        self.documents = read_jsonl(index_dir / "text_chunks.jsonl")
        self.documents += read_jsonl(index_dir / "table_pages.jsonl")
        self.documents += read_jsonl(index_dir / "visual_pages.jsonl")
        self.documents += read_jsonl(index_dir / "visual_notes.jsonl")
        self._tokens = [tokens(x.get("text", "")) for x in self.documents]
        self._avgdl = sum(map(len, self._tokens)) / max(1, len(self._tokens))
        self._df: Counter[str] = Counter()
        for row in self._tokens:
            self._df.update(set(row))

    def search(self, query: str, *, filters=None, top_k: int = 6) -> list[dict]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        q = tokens(query)
        if not q:
            return []
        n = len(self.documents)
        k1, b = 1.4, 0.72
        ranked = []
        if filters and not isinstance(filters, dict):
            raise ValueError("filters must be a mapping of indexed fields to exact values")
        def matches_filters(document: dict) -> bool:
            metadata = document.get("metadata", {})
            allowed = {"source_id", "pdf_page", "printed_page", "kind", *metadata.keys()}
            unknown = set(filters or {}) - allowed
            if unknown:
                raise ValueError(f"Unsupported metadata filter(s): {', '.join(sorted(unknown))}")
            return all(document.get(key, metadata.get(key)) == value for key, value in (filters or {}).items())

        eligible = [i for i, document in enumerate(self.documents) if matches_filters(document)]
        refs = [re.sub(r"\s+", " ", x).casefold() for x in re.findall(r"\b(?:table|figure|box)\s*\d+(?:\.\d+)*", query, flags=re.I)]
        def exact_ref(ref: str, text: str) -> bool:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(ref)}(?!\.\d|\d)", re.I)
            return bool(pattern.search(text))
        ref_pages = {
            self.documents[i].get("pdf_page") for i in eligible
            for row in [self.documents[i]]
            if row.get("kind") in ("visual_page", "table_page")
            and any(exact_ref(ref, " ".join(row.get("captions", [])).casefold()) for ref in refs)
        }
        years = re.findall(r"\b20\d{2}\b", query)
        asks_target = any(term in q for term in ("target", "goal", "objective"))
        for i in eligible:
            doc_tokens = self._tokens[i]
            freqs = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for term in q:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(1, self._avgdl)))
            if refs:
                captions = " ".join(self.documents[i].get("captions", [])).casefold()
                record_text = self.documents[i].get("text", "").casefold()
                # Prefer the compact table-region record over the visual-page caption.
                # The latter identifies a figure/table but does not carry its values.
                exact_caption_refs = sum(exact_ref(ref, captions) for ref in refs)
                if self.documents[i].get("kind") == "table_page":
                    score += 90.0 * exact_caption_refs
                else:
                    score += 4.0 * exact_caption_refs
                if self.documents[i].get("pdf_page") in ref_pages:
                    kind = self.documents[i].get("kind")
                    score += 50.0 if kind == "human_visual_note" else 10.0
                score += 4.0 * sum(exact_ref(ref, record_text[:400]) and not exact_ref(ref, captions) for ref in refs)
            if asks_target and years:
                source_text = self.documents[i].get("text", "").casefold()
                if any(year in source_text for year in years) and re.search(r"\b(?:target|goal|objective)\b", source_text):
                    score += 16.0
            if score <= 0:
                continue
            row = self.documents[i]
            result = {
                "id": row["id"], "text": row["text"], "source_id": row["source_id"],
                "source": row["source"], "page": row.get("printed_page"),
                "metadata": {**row.get("metadata", {}), "pdf_page": row.get("pdf_page"),
                             "kind": row.get("kind"), "crop_path": row.get("crop_path")},
                "score": score,
            }
            ranked.append((score, result))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [row for _, row in ranked[:top_k]]


def persist(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--retrieval-method", choices=["bm25", "dense_cosine", "hybrid_rrf"], default="bm25")
    p.add_argument("--provider", choices=["openai", "groq", "gemini"], default="openai")
    p.add_argument("--fallback-provider", choices=["openai", "groq", "gemini"])
    p.add_argument("--fallback-model")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--budget", type=int, default=5200)
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--limit", type=int, help="Smoke-run only the first N cases; never use partial runs as benchmark scores.")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--output", type=Path, default=ADV / "results")
    args = p.parse_args()

    benchmark_path = ADV / "benchmark" / "questions.json"
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    selection_path = ADV / "benchmark" / "selection.json"
    if args.split == "test":
        if not selection_path.exists():
            raise ValueError("Lock the pipeline on dev in advanced/benchmark/selection.json before running test.")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("benchmark_sha256") != benchmark_hash:
            raise ValueError("The advanced benchmark changed after dev selection; do not run this test set.")
        if selection.get("retrieval_method", "bm25") != args.retrieval_method:
            raise ValueError("The selected retrieval method differs from dev; rerun and review dev before test.")
        expected_pipeline = {
            "bm25": "bm25-text-plus-human-visual-notes",
            "dense_cosine": "embeddinggemma2-dense-cosine",
            "hybrid_rrf": "embeddinggemma2-hybrid-rrf",
        }[args.retrieval_method]
        if selection.get("pipeline") != expected_pipeline:
            raise ValueError("The selected retrieval pipeline is inconsistent with its method.")

    load_dotenv(args.env_file, override=False)
    import os
    provider = args.provider
    api_key_env = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
    model_env = {"openai": "OPENAI_MODEL", "groq": "GROQ_MODEL", "gemini": "GEMINI_MODEL"}
    model_default = {"openai": "gpt-5.4-mini", "groq": "openai/gpt-oss-120b", "gemini": "gemini-3.5-flash-lite"}
    controller_types = {"openai": OpenAIController, "groq": GroqController, "gemini": GeminiController}
    api_key = os.getenv(api_key_env[provider])
    model = os.getenv(model_env[provider]) or model_default[provider]
    if not api_key:
        raise ValueError(f"Set {provider.upper()}_API_KEY in the project .env file.")
    fallback_provider = args.fallback_provider
    fallback_model = args.fallback_model
    if fallback_provider and fallback_model is None:
        fallback_model = os.getenv(model_env[fallback_provider]) or model_default[fallback_provider]
    if args.split == "test" and (
        selection.get("provider") != provider or selection.get("model") != model
        or selection.get("retrieval_method", "bm25") != args.retrieval_method
        or selection.get("fallback_provider") != fallback_provider
        or selection.get("fallback_model") != fallback_model
        or selection.get("top_k") != args.top_k or selection.get("budget") != args.budget
        or selection.get("max_searches") != args.max_searches
    ):
        raise ValueError("The selected route/settings differ from dev; rerun and review dev before test.")
    questions = [x for x in json.loads(benchmark_path.read_text(encoding="utf-8")) if x["split"] == args.split]
    if args.limit:
        questions = questions[:args.limit]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / f"{args.split}_traces.json"
    traces = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    index_digest = hashlib.sha256()
    for name in ["text_chunks.jsonl", "table_pages.jsonl", "visual_pages.jsonl", "visual_notes.jsonl"]:
        path = ADV / "index" / name
        if path.exists():
            index_digest.update(path.read_bytes())
    index_hash = index_digest.hexdigest()
    done = {x["query_id"] for x in traces if x.get("split") == args.split and x.get("model") == model
            and x.get("index_sha256") == index_hash
            and x.get("retrieval_method", "bm25") == args.retrieval_method
            and x.get("status") in ("answered", "abstain")}
    bm25_retriever = OecdRetriever(ADV / "index")
    if args.retrieval_method == "bm25":
        retriever = bm25_retriever
    else:
        from embedding_retriever import DenseOecdRetriever, HybridOecdRetriever, LocalEmbeddingGemma2
        embedder = LocalEmbeddingGemma2()
        dense_retriever = DenseOecdRetriever(ADV / "index", bm25_retriever, embedder)
        retriever = dense_retriever if args.retrieval_method == "dense_cosine" else HybridOecdRetriever(
            bm25_retriever, dense_retriever
        )

    for case in questions:
        if case["id"] in done:
            continue
        extra_system = (
                "For questions with multiple requested parts, answer every part explicitly and cite the excerpt "
                "that supports each part. Preserve the metric, period, units and numeric values exactly as stated. "
                "If a unit appears in a row label or table heading rather than beside each cell, carry it into the answer; "
                "for example, a row marked ‘% of labour force’ means a cell value of 2.5 must be reported as 2.5%. "
                "Only calculate arithmetic the question directly requests. For figure-number questions, prioritize "
                "evidence from that exact figure's page. When a question asks for a figure's title and subject, state "
                "the exact title and describe the chart type, metric and all series or categories named in the visual note; "
                "avoid vague summaries such as 'compared with peers'. Search with the report's exact terminology when possible."
        )
        controller_type = controller_types[provider]
        controller = controller_type(api_key, model, max_searches=args.max_searches, max_retries=1,
                                     extra_system=extra_system)
        attempts = []
        start = time.perf_counter()
        try:
            result = run_agent(case["question"], controller, retriever, max_searches=args.max_searches,
                               top_k=args.top_k, budget=args.budget)
            error = None
        except Exception as exc:
            result = {"status": "error", "answer": None, "citations": [], "events": []}
            error = f"{type(exc).__name__}: {str(exc).replace(api_key, '[redacted]')}"
        attempts.append({"provider": provider, "model": model, "status": result["status"], "error": error,
                         "usage": controller.usage, "controller_decisions": controller.decisions})
        selected_provider, selected_model = provider, model
        selected_controller = controller
        if fallback_provider and result["status"] != "answered":
            fallback_key = os.getenv(api_key_env[fallback_provider])
            if not fallback_key:
                raise ValueError(f"Set {fallback_provider.upper()}_API_KEY in the project .env file.")
            fallback_type = controller_types[fallback_provider]
            fallback_controller = fallback_type(fallback_key, fallback_model, max_searches=args.max_searches,
                                                max_retries=1, extra_system=extra_system)
            try:
                fallback_result = run_agent(case["question"], fallback_controller, retriever,
                                            max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
                fallback_error = None
            except Exception as exc:
                fallback_result = {"status": "error", "answer": None, "citations": [], "events": []}
                fallback_error = f"{type(exc).__name__}: {str(exc).replace(fallback_key, '[redacted]')}"
            attempts.append({"provider": fallback_provider, "model": fallback_model,
                             "status": fallback_result["status"], "error": fallback_error,
                             "usage": fallback_controller.usage,
                             "controller_decisions": fallback_controller.decisions})
            result = fallback_result
            error = fallback_error
            selected_provider, selected_model = fallback_provider, fallback_model
            selected_controller = fallback_controller
        row = {
            "query_id": case["id"], "split": args.split, "group": case["group"],
            "retrieval_method": args.retrieval_method,
            "question": case["question"], "status": result["status"], "answer": result.get("answer"),
            "citations": result.get("citations", []), "events": result.get("events", []),
            "provider": selected_provider, "model": selected_model, "requested_provider": provider,
            "controller_version": selected_controller.version, "attempts": attempts,
            "index_sha256": index_hash,
            "top_k": args.top_k, "budget": args.budget, "max_searches": args.max_searches,
            "usage": selected_controller.usage, "controller_decisions": selected_controller.decisions,
            "elapsed_seconds": time.perf_counter() - start, "error": error,
        }
        traces = [x for x in traces if x.get("query_id") != case["id"]] + [row]
        persist(trace_path, traces)
        print(f"{case['id']} {row['status']} ({row['elapsed_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
