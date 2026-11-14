"""Create local EmbeddingGemma 2 vectors for all searchable OECD text records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "advanced" / "index"
DEFAULT_MODEL = r"D:\Tuyen\vinuni\lab\K5-L3A-Day05-Track3-ThoCode\models\embeddinggemma-2"
INPUT_FILES = (
    "text_chunks.jsonl",
    "table_pages.jsonl",
    "visual_pages.jsonl",
    "visual_notes.jsonl",
)


def read_documents() -> list[dict]:
    documents = []
    for filename in INPUT_FILES:
        path = INDEX / filename
        if not path.is_file():
            continue
        documents.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    seen: set[str] = set()
    for document in documents:
        doc_id = document.get("id")
        if not doc_id or doc_id in seen:
            raise ValueError(f"Missing or duplicate document ID: {doc_id!r}")
        if not document.get("text", "").strip():
            raise ValueError(f"Document has no searchable text: {doc_id}")
        seen.add(doc_id)
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local EmbeddingGemma 2 model not found: {model_path}")

    documents = read_documents()
    index_hash = hashlib.sha256()
    for filename in INPUT_FILES:
        path = INDEX / filename
        if path.exists():
            index_hash.update(path.read_bytes())
    input_hash = index_hash.hexdigest()

    vector_path = INDEX / "embeddinggemma2_vectors.npz"
    records_path = INDEX / "embeddinggemma2_records.jsonl"
    manifest_path = INDEX / "embeddinggemma2_manifest.json"
    checkpoint_path = INDEX / "embeddinggemma2_checkpoint.json"
    partial_vector_path = INDEX / "embeddinggemma2_vectors.partial.npz"
    partial_records_path = INDEX / "embeddinggemma2_records.partial.jsonl"
    expected_ids = [doc["id"] for doc in documents]
    vectors_done = np.empty((0, 768), dtype=np.float32)
    completed = 0

    # Reuse vectors for unchanged, content-addressed records. This makes adding
    # a few reviewed visual notes inexpensive while preserving the exact model.
    if manifest_path.is_file() and vector_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_model = Path(old_manifest.get("model", "")).resolve() == model_path.resolve()
        if same_model:
            with np.load(vector_path) as saved:
                saved_ids = saved["ids"].astype(str).tolist()
                saved_vectors = np.asarray(saved["vectors"], dtype=np.float32)
            shape_ok = saved_vectors.shape == (len(saved_ids), 768)
            finite_ok = shape_ok and np.isfinite(saved_vectors).all()
            normalized_ok = finite_ok and np.allclose(
                np.linalg.norm(saved_vectors, axis=1), 1.0, rtol=1e-4, atol=1e-4
            )
            if normalized_ok:
                common = 0
                while common < min(len(saved_ids), len(expected_ids)) and saved_ids[common] == expected_ids[common]:
                    common += 1
                if (old_manifest.get("input_index_sha256") == input_hash
                        and common == len(expected_ids) == len(saved_ids)
                        and old_manifest.get("count") == len(expected_ids)):
                    record_ids = [json.loads(line)["id"] for line in records_path.read_text(encoding="utf-8").splitlines()
                                  if line.strip()] if records_path.is_file() else []
                    if record_ids == expected_ids:
                        print(f"Embedding index is current: {len(expected_ids)}/{len(expected_ids)} records", flush=True)
                        return 0
                if common:
                    vectors_done = saved_vectors[:common].copy()
                    completed = common
                    print(f"Reusing {completed}/{len(documents)} unchanged vectors", flush=True)

    print(f"Loading local embedding model: {model_path}", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    model = SentenceTransformer(
        str(model_path),
        local_files_only=True,
        model_kwargs={"dtype": torch.float32},
    )
    dimensions = model.get_sentence_embedding_dimension()
    if dimensions != 768:
        raise ValueError(f"Expected 768-dimensional EmbeddingGemma 2; got {dimensions}")

    batch_size = 8
    if checkpoint_path.exists() and partial_vector_path.exists() and partial_records_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        with np.load(partial_vector_path) as saved:
            saved_ids = saved["ids"].tolist()
            saved_vectors = np.asarray(saved["vectors"], dtype=np.float32)
        expected_ids = [doc["id"] for doc in documents[:len(saved_ids)]]
        if (checkpoint.get("input_index_sha256") == input_hash
                and saved_ids == expected_ids
                and len(saved_ids) == checkpoint.get("completed")
                and len(partial_records_path.read_text(encoding="utf-8").splitlines()) == len(saved_ids)):
            if len(saved_ids) > completed:
                vectors_done = saved_vectors
                completed = len(saved_ids)
                print(f"Resuming from checkpoint: {completed}/{len(documents)}", flush=True)

    for start in range(completed, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        batch = documents[start:end]
        batch_vectors = np.asarray(
            model.encode(
                [document["text"] for document in batch],
                prompt_name="Document",
                batch_size=batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True).clip(min=1e-12)
        batch_vectors /= norms
        vectors_done = np.concatenate((vectors_done, batch_vectors), axis=0)
        completed = end

        partial_ids = np.asarray([doc["id"] for doc in documents[:completed]])
        vector_tmp = partial_vector_path.with_suffix(".npz.tmp")
        with vector_tmp.open("wb") as handle:
            np.savez_compressed(handle, ids=partial_ids, vectors=vectors_done)
        vector_tmp.replace(partial_vector_path)
        partial_records = [
            {
                "id": doc["id"],
                "source_id": doc.get("source_id"),
                "source": doc.get("source"),
                "page": doc.get("printed_page"),
                "metadata": {
                    **doc.get("metadata", {}),
                    "pdf_page": doc.get("pdf_page"),
                    "kind": doc.get("kind"),
                    "crop_path": doc.get("crop_path"),
                },
            }
            for doc in documents[:completed]
        ]
        records_tmp = partial_records_path.with_suffix(".jsonl.tmp")
        records_tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in partial_records),
            encoding="utf-8",
        )
        records_tmp.replace(partial_records_path)
        checkpoint_tmp = checkpoint_path.with_suffix(".json.tmp")
        checkpoint_tmp.write_text(
            json.dumps({"input_index_sha256": input_hash, "completed": completed}, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint_tmp.replace(checkpoint_path)
        print(f"Embedded {completed}/{len(documents)} searchable records", flush=True)

    vectors = vectors_done
    records = [
        {
            "id": doc["id"],
            "source_id": doc.get("source_id"),
            "source": doc.get("source"),
            "page": doc.get("printed_page"),
            "metadata": {
                **doc.get("metadata", {}),
                "pdf_page": doc.get("pdf_page"),
                "kind": doc.get("kind"),
                "crop_path": doc.get("crop_path"),
            },
        }
        for doc in documents
    ]
    vector_tmp = vector_path.with_suffix(".npz.tmp")
    with vector_tmp.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray([d["id"] for d in documents]), vectors=vectors)
    vector_tmp.replace(vector_path)
    records_tmp = records_path.with_suffix(".jsonl.tmp")
    records_tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    records_tmp.replace(records_path)
    manifest = {
        "model": str(model_path),
        "dimensions": dimensions,
        "document_prompt": "Document",
        "normalization": "L2; cosine-compatible",
        "count": len(documents),
        "counts_by_kind": dict(Counter(doc.get("kind", "unknown") for doc in documents)),
        "input_index_sha256": index_hash.hexdigest(),
        "vectors_file": vector_path.name,
        "records_file": records_path.name,
        "query_prompt": "SearchQuery",
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    for path in (checkpoint_path, partial_vector_path, partial_records_path):
        path.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
