"""Run real agent cases and switch provider when a rate limit is reached."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from day6_retrieval import Day6Retriever, DEFAULT_STORE, DEFAULT_MODEL
from gemini_controller import GeminiController
from groq_controller import GroqController
from agent_runtime import AgentRunError, available_providers, run_with_fallback
from retrieval_benchmark import CONFIGS, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev','test'], required=True)
    p.add_argument('--configs', nargs='+', choices=list(CONFIGS), required=True)
    p.add_argument('--max-searches', type=int, default=2)
    p.add_argument('--output', type=Path, default=ROOT/'starter'/'results'/'agent')
    p.add_argument('--benchmark', type=Path, default=ROOT/'data'/'benchmark_day7.json')
    p.add_argument('--env-file', type=Path, default=ROOT/'.env')
    p.add_argument('--store', type=Path, default=DEFAULT_STORE)
    p.add_argument('--embedding-model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--provider', choices=['gemini','groq'], default='gemini')
    args = p.parse_args()
    if args.split == 'test':
        selection_path = ROOT/'starter'/'results'/'retrieval'/'selection.json'
        if not selection_path.exists(): raise ValueError('Record selection on dev before running test')
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if selection.get('benchmark_sha256') != hashlib.sha256(args.benchmark.read_bytes()).hexdigest():
            raise ValueError('Benchmark changed after configuration selection')
        if not set(args.configs).issubset({selection['baseline'], selection['chosen']}):
            raise ValueError('Only the preselected baseline and configuration can run on test')
    load_dotenv(args.env_file, override=False)
    providers = available_providers()
    if args.provider not in providers: raise ValueError(f'Missing key for {args.provider}')
    current_provider = args.provider
    questions = [q for q in json.loads(args.benchmark.read_text(encoding='utf-8')) if q['split']==args.split]
    retriever = Day6Retriever('bm25',args.store,args.embedding_model)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/f'{args.split}_traces.json'
    traces = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    models = {item[2] for item in providers.values()}
    versions = {GeminiController.version,GroqController.version}
    completed = {(x['configuration'], x['query_id'], x['question'], x.get('max_searches')) for x in traces
                 if x['status'] in ('answered', 'abstain') and x.get('controller_version') in versions and x.get('model') in models}
    for name in args.configs:
        method, top_k, budget = CONFIGS[name]
        retriever.method = method
        for q in questions:
            if (name, q['id'], q['question'], args.max_searches) in completed: continue
            start = time.perf_counter()
            controller, used_provider, attempts = None, current_provider, []
            prior = next((x for x in traces if x['configuration']==name and x['query_id']==q['id']
                          and x.get('question')==q['question'] and x['status']=='error'), None)
            try:
                result, controller, used_provider, attempts = run_with_fallback(q['question'], retriever,
                    provider=current_provider, top_k=top_k, budget=budget, max_searches=args.max_searches)
                current_provider = used_provider
            except Exception as exc:
                if isinstance(exc, AgentRunError):
                    attempts = exc.attempts
                    if attempts: used_provider = attempts[-1]['provider']
                result = {'status':'error','error':f'{type(exc).__name__}: {exc}','events':[],'citations':[]}
            if prior: attempts = prior.get('failed_attempts',[]) + attempts
            row = {'query_id': q['id'], 'question': q['question'], 'configuration':name,
                   'provider':used_provider,'model':controller.model if controller else None,
                   'top_k':top_k,'budget':budget,'max_searches':args.max_searches,
                   'elapsed_seconds':time.perf_counter()-start,'usage':controller.usage if controller else [],
                   'failed_attempts':attempts,'controller_version':controller.version if controller else None,
                   'controller_decisions':controller.decisions if controller else [],**result}
            traces = [x for x in traces if (x['configuration'],x['query_id']) != (name,q['id'])] + [row]
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            temporary.replace(path)
            print(f"{name} {q['id']}: {row['status']} ({used_provider}; failed attempts: {len(attempts)})",flush=True)
            if row['status']=='error':
                print(f"Stopped after error: {row['error']}",flush=True)
                raise SystemExit(1)


if __name__ == '__main__': main()
