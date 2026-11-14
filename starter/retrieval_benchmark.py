"""Measure retrieval and context evidence on a fixed benchmark split."""
import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path
from baseline import pack_context
from day6_retrieval import Day6Retriever, DEFAULT_STORE, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'dense-3-800': ('dense', 3, 800),
    'bm25-3-800': ('bm25', 3, 800),
    'hybrid-3-800': ('hybrid', 3, 800),
    'hybrid-5-2400': ('hybrid', 5, 2400),
    'hybrid-8-4000': ('hybrid', 8, 4000),
}


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def hit(chunk, gold):
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page']
               and norm(g['evidence']) in norm(chunk['text']) for g in gold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--benchmark', type=Path, default=ROOT/'data'/'benchmark_day7.json')
    parser.add_argument('--output', type=Path, default=ROOT/'starter'/'results'/'retrieval')
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--embedding-model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    configs = CONFIGS
    if args.split == 'test':
        selection_path = args.output/'selection.json'
        if not selection_path.is_file():
            raise ValueError('Record the selected dev configuration in selection.json before test')
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if selection.get('benchmark_sha256') != hashlib.sha256(args.benchmark.read_bytes()).hexdigest():
            raise ValueError('Benchmark changed after configuration selection')
        names = (selection['baseline'], selection['chosen'])
        configs = {name: CONFIGS[name] for name in dict.fromkeys(names)}
    questions = [q for q in json.loads(args.benchmark.read_text(encoding='utf-8')) if q['split'] == args.split]
    retriever = Day6Retriever('dense',args.store,args.embedding_model)
    traces, summary = [], []
    warmup_start = time.perf_counter()
    retriever._vector(questions[0]['question'])
    warmup_seconds = time.perf_counter() - warmup_start
    retriever._query_vectors.clear()
    for q in questions:
        vector_start = time.perf_counter()
        retriever._vector(q['question'])
        vector_seconds = time.perf_counter() - vector_start
        for name, (method, top_k, budget) in configs.items():
            retriever.method = method
            start = time.perf_counter()
            results = retriever.search(q['question'], top_k=max(top_k, 3))
            elapsed = time.perf_counter() - start + (vector_seconds if method in ('dense', 'hybrid') else 0)
            context, ids = pack_context(results[:top_k], budget=budget)
            is_answerable = q.get('answerable', True)
            gold = q.get('gold', [])
            traces.append({'query_id': q['id'], 'configuration': name, 'answerable': is_answerable,
                'query': q['question'], 'filter': None, 'results': results, 'context': context,
                'context_ids': ids, 'hit1': any(hit(c, gold) for c in results[:1]) if is_answerable else None,
                'hit3': any(hit(c, gold) for c in results[:3]) if is_answerable else None,
                'context_hit': any(hit(c, gold) for c in results if c['id'] in ids) if is_answerable else None,
                'elapsed_seconds': elapsed, 'embedding_seconds': vector_seconds if method in ('dense', 'hybrid') else 0})
    for name in configs:
        rows = [t for t in traces if t['configuration'] == name and t['answerable']]
        n = len(rows)
        summary.append({'configuration': name, 'n_answerable': n,
                        **{key: {'hits': sum(bool(t[key]) for t in rows), 'total': n,
                                 'rate': sum(bool(t[key]) for t in rows)/n if n else None}
                           for key in ('hit1','hit3','context_hit')},
                        'mean_retrieval_seconds': sum(t['elapsed_seconds'] for t in rows)/n if n else None,
                        'embedding_cost': 0, 'model_warmup_seconds': warmup_seconds})
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in [(f'{args.split}_traces.json', traces), (f'{args.split}_summary.json', summary)]:
        (args.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
