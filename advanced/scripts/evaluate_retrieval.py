"""Compare OECD retrieval modes on the assignment's published dev split only."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADV = ROOT / "advanced"
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ADV / "scripts"))

from baseline import pack_context  # noqa: E402
from embedding_retriever import (  # noqa: E402
    DenseOecdRetriever,
    HybridOecdRetriever,
    LocalEmbeddingGemma2,
)
from run_advanced import OecdRetriever, read_jsonl  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def supports(hit: dict, gold: dict) -> bool:
    return (
        hit.get("source_id") == gold["source_id"]
        and hit.get("metadata", {}).get("pdf_page") == gold["pdf_page"]
        and normalize(gold["evidence"]) in normalize(hit.get("text", ""))
    )


def evidence_indices(hits: list[dict], gold: list[dict]) -> list[list[int]]:
    return [[i for i, hit in enumerate(hits, 1) if supports(hit, item)] for item in gold]


def persist(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    index_dir = ADV / "index"
    benchmark_path = ADV / "benchmark" / "questions.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dev = [case for case in benchmark if case.get("split") == "dev"]
    answerable = [case for case in dev if case.get("answerable")]
    unanswerable = [case for case in dev if not case.get("answerable")]
    if (len(dev), len(answerable), len(unanswerable)) != (18, 15, 3):
        raise ValueError("Expected the published advanced dev split: 15 answerable and 3 unanswerable cases")

    records = sum((read_jsonl(index_dir / name) for name in (
        "text_chunks.jsonl", "table_pages.jsonl", "visual_pages.jsonl", "visual_notes.jsonl"
    )), [])
    indexed_by_page = defaultdict(list)
    for row in records:
        indexed_by_page[(row.get("source_id"), row.get("pdf_page"))].append(row.get("text", ""))
    for case in answerable:
        for gold in case.get("gold", []):
            if not any(normalize(gold["evidence"]) in normalize(text)
                       for text in indexed_by_page[(gold["source_id"], gold["pdf_page"])]):
                raise ValueError(f"Gold evidence is not present in the current source-page index: {case['id']}")

    bm25 = OecdRetriever(index_dir)
    embedder = LocalEmbeddingGemma2()
    dense = DenseOecdRetriever(index_dir, bm25, embedder)
    methods = {
        "bm25": bm25,
        "dense_cosine": dense,
        "hybrid_rrf": HybridOecdRetriever(bm25, dense),
    }
    traces: list[dict] = []
    totals = {name: defaultdict(list) for name in methods}
    for case in dev:
        for name, retriever in methods.items():
            start = time.perf_counter()
            top_hits = retriever.search(case["question"], top_k=6)
            elapsed = time.perf_counter() - start
            context, context_ids = pack_context(top_hits, budget=5200)
            gold = case.get("gold", []) if case["answerable"] else []
            locations = evidence_indices(top_hits, gold)
            top3_locations = evidence_indices(top_hits[:3], gold)
            context_coverage = [
                any(hit.get("id") in context_ids and supports(hit, item) for hit in top_hits)
                for item in gold
            ]
            hit1_any = bool(locations and any(1 in ranks for ranks in locations))
            hit3_any = bool(locations and any(any(rank <= 3 for rank in ranks) for ranks in locations))
            complete3 = bool(locations and all(ranks for ranks in top3_locations))
            context_complete = bool(context_coverage and all(context_coverage))
            trace = {
                "query_id": case["id"], "group": case["group"],
                "answerable": case["answerable"], "method": name,
                "question": case["question"], "gold": gold,
                "top_hits": [
                    {"rank": rank, "id": hit["id"], "pdf_page": hit.get("metadata", {}).get("pdf_page"),
                     "kind": hit.get("metadata", {}).get("kind"), "score": hit["score"],
                     "evidence_indices": [i for i, item in enumerate(gold) if supports(hit, item)]}
                    for rank, hit in enumerate(top_hits, 1)
                ],
                "context_ids": context_ids, "context_chars": len(context),
                "hit1_any_evidence": hit1_any, "evidence_hit3_any": hit3_any,
                "complete_evidence_at_3": complete3, "context_has_complete_evidence": context_complete,
                "evidence_ranks": locations, "context_coverage_by_gold": context_coverage,
                "elapsed_seconds": elapsed,
            }
            traces.append(trace)
            if case["answerable"]:
                totals[name]["hit1"].append(hit1_any)
                totals[name]["hit3"].append(hit3_any)
                totals[name]["complete3"].append(complete3)
                totals[name]["context"].append(context_complete)
                totals[name]["elapsed"].append(elapsed)
                totals[name][f"group:{case['group']}:hit3"].append(hit3_any)
                totals[name][f"group:{case['group']}:context"].append(context_complete)
            print(f"{case['id']} {name} {'hit@3' if hit3_any else 'miss@3'} ({elapsed:.3f}s)", flush=True)

    summary = {
        "split": "dev", "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "methods": {}, "metric_definitions": {
            "hit1_any_evidence": "At least one exact gold evidence passage is ranked first.",
            "evidence_hit3_any": "At least one exact gold evidence passage is present in the top three.",
            "complete_evidence_at_3": "Every required gold evidence passage is present in the top three.",
            "context_evidence_rate": "Every required gold passage is included in pack_context(top-6, 5200 characters).",
            "answerable_denominator": len(answerable),
        },
        "configuration": {"top_k": 6, "context_budget_chars": 5200,
                          "embedding_model": embedder.model_path, "document_prompt": "Document",
                          "query_prompt": "SearchQuery", "dimensions": 768},
    }
    for name, metrics in totals.items():
        summary["methods"][name] = {
            "hit1_any_evidence": {"n": sum(metrics["hit1"]), "N": len(metrics["hit1"]),
                                  "rate": sum(metrics["hit1"]) / len(metrics["hit1"])},
            "evidence_hit3_any": {"n": sum(metrics["hit3"]), "N": len(metrics["hit3"]),
                                  "rate": sum(metrics["hit3"]) / len(metrics["hit3"])},
            "complete_evidence_at_3": {"n": sum(metrics["complete3"]), "N": len(metrics["complete3"]),
                                       "rate": sum(metrics["complete3"]) / len(metrics["complete3"])},
            "context_evidence_rate": {"n": sum(metrics["context"]), "N": len(metrics["context"]),
                                      "rate": sum(metrics["context"]) / len(metrics["context"])},
            "mean_retrieval_seconds": statistics.mean(metrics["elapsed"]),
            "by_group": {
                group: {
                    "evidence_hit3": {"n": sum(metrics[f"group:{group}:hit3"]),
                                      "N": len(metrics[f"group:{group}:hit3"])},
                    "context_complete": {"n": sum(metrics[f"group:{group}:context"]),
                                         "N": len(metrics[f"group:{group}:context"])},
                }
                for group in sorted({case["group"] for case in answerable})
            },
        }
    output = ADV / "results" / "retrieval-dev-embedding"
    persist(output / "summary.json", summary)
    persist(output / "traces.json", traces)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved dev retrieval evidence to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
