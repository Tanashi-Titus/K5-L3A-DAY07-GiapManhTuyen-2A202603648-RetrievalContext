"""Ask a question using the Day 6 index and the OpenAI GPT controller."""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from day6_retrieval import DAY6, DEFAULT_MODEL, DEFAULT_STORE, Day6Retriever
from agent_runtime import run_with_fallback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('question')
    p.add_argument('--method', choices=['dense', 'bm25', 'hybrid'], default='hybrid')
    p.add_argument('--store', type=Path, default=DEFAULT_STORE)
    p.add_argument('--embedding-model', type=Path, default=DEFAULT_MODEL)
    local_env = Path(__file__).resolve().parents[1] / '.env'
    p.add_argument('--env-file', type=Path, default=local_env if local_env.exists() else DAY6 / 'starter' / '.env')
    p.add_argument('--model', default=None)
    p.add_argument('--provider', choices=['openai','gemini','groq'], default='openai')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--budget', type=int, default=1600)
    p.add_argument('--max-searches', type=int, default=3)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    load_dotenv(args.env_file, override=False)
    if args.model: os.environ[args.provider.upper()+'_MODEL'] = args.model
    retriever = Day6Retriever(args.method, args.store, args.embedding_model)
    start = time.perf_counter()
    result, controller, provider, attempts = run_with_fallback(args.question, retriever, provider=args.provider,
                       top_k=args.top_k, budget=args.budget, max_searches=args.max_searches)
    hits = {chunk['id']: chunk for event in result['events'] if event.get('event') == 'search'
            for chunk in event['results']}
    sources = [{'id': id_, 'source_id': hits[id_]['source_id'], 'source': hits[id_]['source'],
                'page': hits[id_]['page']} for id_ in result['citations'] if id_ in hits]
    result.update({'question': args.question, 'method': args.method, 'provider': provider, 'model': controller.model,
                   'sources': sources,
                   'usage': controller.usage, 'controller_decisions': controller.decisions, 'failed_attempts': attempts,
                   'elapsed_seconds': time.perf_counter() - start})
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(json.dumps({'status':result['status'], 'provider':result['provider'], 'model':result['model'],
                      'answer':result['answer'], 'sources':result['sources'],
                      'trace_file':str(args.output) if args.output else None},ensure_ascii=False,indent=2) if args.output else rendered)


if __name__ == '__main__':
    main()
