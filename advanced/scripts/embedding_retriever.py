"""Dense cosine and BM25+dense RRF retrieval for the OECD index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = r"D:\Tuyen\vinuni\lab\K5-L3A-Day05-Track3-ThoCode\models\embeddinggemma-2"
INPUT_FILES = ("text_chunks.jsonl", "table_pages.jsonl", "visual_pages.jsonl", "visual_notes.jsonl")


def matches_filters(document: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    if not isinstance(filters, dict):
        raise ValueError("filters must be a mapping of indexed fields to exact values")
    metadata = document.get("metadata", {})
    allowed = {"source_id", "pdf_page", "printed_page", "kind", *metadata.keys()}
    unknown = set(filters) - allowed
    if unknown:
        raise ValueError(f"Unsupported metadata filter(s): {', '.join(sorted(unknown))}")
    return all(document.get(key, metadata.get(key)) == value for key, value in filters.items())


class LocalEmbeddingGemma2:
    def __init__(self, model_path: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"EmbeddingGemma 2 model not found: {self.model_path}")
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        self.model = SentenceTransformer(
            self.model_path,
            local_files_only=True,
            model_kwargs={"dtype": torch.float32},
        )
        if self.model.get_sentence_embedding_dimension() != 768:
            raise ValueError("Expected 768-dimensional EmbeddingGemma 2 vectors")

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.asarray(
            self.model.encode(query, prompt_name="SearchQuery", normalize_embeddings=False),
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Query embedding is non-finite or has zero norm")
        return vector / norm


class DenseOecdRetriever:
    def __init__(self, index_dir: Path, bm25_retriever, embedder: LocalEmbeddingGemma2):
        self.index_dir = index_dir
        self.bm25 = bm25_retriever
        self.embedder = embedder
        manifest_path = index_dir / "embeddinggemma2_manifest.json"
        vectors_path = index_dir / "embeddinggemma2_vectors.npz"
        if not manifest_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError("Run advanced/scripts/embed_index.py before dense retrieval")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        for filename in INPUT_FILES:
            path = index_dir / filename
            if path.exists():
                digest.update(path.read_bytes())
        if digest.hexdigest() != manifest.get("input_index_sha256"):
            raise ValueError("Embedding vectors are stale; rebuild them from the current OECD index")
        if Path(manifest.get("model", "")).resolve() != Path(embedder.model_path).resolve():
            raise ValueError("Embedding model does not match the model recorded in the dense index manifest")
        with np.load(vectors_path) as saved:
            self.ids = saved["ids"].astype(str)
            self.vectors = np.asarray(saved["vectors"], dtype=np.float32)
        if manifest.get("count") != len(self.ids) or self.vectors.shape != (len(self.ids), 768):
            raise ValueError(f"Invalid dense index shape: {self.vectors.shape}")
        self.documents = {doc["id"]: doc for doc in self.bm25.documents}
        expected_ids = [doc["id"] for doc in self.bm25.documents]
        if self.ids.tolist() != expected_ids:
            raise ValueError("Embedding IDs or vector order do not match the current searchable records")
        records_path = index_dir / manifest.get("records_file", "")
        if not records_path.is_file():
            raise ValueError("Embedding metadata records are missing")
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row.get("id") for row in records] != expected_ids:
            raise ValueError("Embedding metadata IDs or order do not match the vector index")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Dense index contains non-finite vector values")
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("Dense index contains vectors that are not L2-normalized")

    def search(self, query: str, *, filters=None, top_k: int = 6) -> list[dict]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        query_vector = self.embedder.embed_query(query)
        scores = self.vectors @ query_vector
        eligible = np.asarray([
            matches_filters(self.documents[str(doc_id)], filters) for doc_id in self.ids
        ], dtype=bool)
        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            return []
        count = min(top_k, len(eligible_indices))
        eligible_scores = scores[eligible_indices]
        candidates = np.argpartition(eligible_scores, -count)[-count:]
        candidates = eligible_indices[candidates]
        ranked = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), str(self.ids[i])))
        output = []
        for index in ranked:
            document = self.documents[str(self.ids[index])]
            output.append({
                "id": document["id"],
                "text": document["text"],
                "source_id": document["source_id"],
                "source": document["source"],
                "page": document.get("printed_page"),
                "metadata": {
                    **document.get("metadata", {}),
                    "pdf_page": document.get("pdf_page"),
                    "kind": document.get("kind"),
                    "crop_path": document.get("crop_path"),
                },
                "score": float(scores[index]),
            })
        return output


class HybridOecdRetriever:
    """Reciprocal-rank fusion of BM25 and EmbeddingGemma dense retrieval."""

    def __init__(self, bm25_retriever, dense_retriever: DenseOecdRetriever, rrf_constant: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_constant = rrf_constant

    def search(self, query: str, *, filters=None, top_k: int = 6) -> list[dict]:
        lexical = self.bm25.search(query, filters=filters, top_k=top_k)
        semantic = self.dense.search(query, filters=filters, top_k=top_k)
        fused: dict[str, dict] = {}
        for result_set, source_name in ((lexical, "bm25_score"), (semantic, "dense_score")):
            for rank, result in enumerate(result_set, start=1):
                item = fused.setdefault(result["id"], {**result, "score": 0.0})
                item[source_name] = result["score"]
                item["score"] += 1.0 / (self.rrf_constant + rank)
        return sorted(fused.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]
