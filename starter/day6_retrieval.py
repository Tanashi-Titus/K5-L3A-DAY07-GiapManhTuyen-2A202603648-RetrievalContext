"""Retrieve from the Day 6 Chroma collection with dense, BM25 or RRF search."""
import json
import os
import re
from pathlib import Path

DAY6 = Path(r'D:\Tuyen\vinuni\lab\K5-L3A-DAY06-GiapManhTuyen-2A202603648-DataIngestion')
DEFAULT_STORE = DAY6 / 'results' / 'chroma'
DEFAULT_MODEL = Path(r'D:\Tuyen\vinuni\lab\K5-L3A-Day05-Track3-ThoCode\models\embeddinggemma-2')


def tokenize(text):
    return re.findall(r'\w+', text.casefold())


class Day6Retriever:
    def __init__(self, method='dense', store=DEFAULT_STORE, model=DEFAULT_MODEL, collection='day06-c', candidates=20):
        if method not in ('dense', 'bm25', 'hybrid'):
            raise ValueError('method must be dense, bm25 or hybrid')
        import chromadb
        from rank_bm25 import BM25Okapi
        self.method, self.model_path, self.candidates = method, Path(model), candidates
        if not (Path(store) / 'chroma.sqlite3').is_file():
            raise FileNotFoundError(f'Day 6 Chroma database not found: {store}')
        self.client = chromadb.PersistentClient(path=str(store))
        self.collection = self.client.get_collection(collection)
        raw = self.collection.get(include=['documents', 'metadatas'])
        self.chunks = [self._chunk(i, t, m) for i, t, m in zip(raw['ids'], raw['documents'], raw['metadatas'])]
        if not self.chunks:
            raise ValueError('Day 6 collection is empty')
        self.bm25 = BM25Okapi([tokenize(c['text']) for c in self.chunks])
        self.embedder = None
        self._query_vectors = {}

    @staticmethod
    def _chunk(id_, text, metadata):
        metadata = dict(metadata)
        extra = json.loads(metadata.pop('extra_json', '{}'))
        return {'id': id_, 'text': text, 'source_id': metadata['document_id'],
                'source': metadata['source'], 'page': metadata['page'], 'metadata': extra}

    def _vector(self, query):
        if query in self._query_vectors:
            return self._query_vectors[query]
        if self.embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            if not self.model_path.is_dir():
                raise FileNotFoundError(f'Embedding model not found: {self.model_path}')
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            self.embedder = SentenceTransformer(str(self.model_path), local_files_only=True,
                                                model_kwargs={'dtype': torch.float32})
            if self.embedder.get_sentence_embedding_dimension() != 768:
                raise ValueError('The Day 6 index requires 768-dimensional embeddings')
        import numpy as np
        vector = np.asarray(self.embedder.encode(query, prompt_name='SearchQuery', normalize_embeddings=False), dtype=np.float32)
        result = (vector / max(float(np.linalg.norm(vector)), 1e-12)).tolist()
        self._query_vectors[query] = result
        return result

    def _dense(self, query, top_k, filters):
        where = None
        if filters:
            if not set(filters).issubset({'section', 'year', 'document_type', 'version'}):
                raise ValueError('Unsupported metadata filter')
            clauses = [{k: {'$eq': v}} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {'$and': clauses}
        r = self.collection.query(query_embeddings=[self._vector(query)], n_results=min(top_k, len(self.chunks)), where=where,
                                  include=['documents', 'metadatas', 'distances'])
        return [{**self._chunk(i, t, m), 'score': 1 - d} for i, t, m, d in
                zip(r['ids'][0], r['documents'][0], r['metadatas'][0], r['distances'][0])]

    def _lexical(self, query, top_k, filters):
        if filters and not set(filters).issubset({'section', 'year', 'document_type', 'version'}):
            raise ValueError('Unsupported metadata filter')
        scores = self.bm25.get_scores(tokenize(query))
        rows = [{**c, 'score': float(score)} for c, score in zip(self.chunks, scores)
                if score > 0 and all(c['metadata'].get(k) == v for k, v in (filters or {}).items())]
        return sorted(rows, key=lambda c: (-c['score'], c['id']))[:top_k]

    def search(self, query, *, filters=None, top_k=3):
        if not query.strip() or top_k < 1:
            raise ValueError('query and top_k must be valid')
        if self.method == 'dense':
            return self._dense(query, top_k, filters)
        if self.method == 'bm25':
            return self._lexical(query, top_k, filters)
        rankings = (self._dense(query, max(top_k, self.candidates), filters),
                    self._lexical(query, max(top_k, self.candidates), filters))
        fused = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, 1):
                entry = fused.setdefault(chunk['id'], {**chunk, 'score': 0.0})
                entry['score'] += 1 / (60 + rank)
        return sorted(fused.values(), key=lambda c: (-c['score'], c['id']))[:top_k]
