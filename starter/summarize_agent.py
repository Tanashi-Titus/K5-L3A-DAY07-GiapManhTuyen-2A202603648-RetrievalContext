"""Aggregate an individual configuration after its reviews are complete."""
import argparse
import json
import statistics
from pathlib import Path
from agent_metrics import aggregate
from retrieval_benchmark import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev','test'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--results-root', type=Path, default=ROOT/'starter'/'results')
    args = p.parse_args()
    cases=[c for c in json.loads((ROOT/'data'/'benchmark_day7.json').read_text(encoding='utf-8')) if c['split']==args.split]
    traces=[t for t in json.loads((args.results_root/'agent'/f'{args.split}_traces.json').read_text(encoding='utf-8')) if t['configuration']==args.config]
    reviews=json.loads(args.reviews.read_text(encoding='utf-8'))
    summary=aggregate(cases,traces,reviews)
    counts=[sum(e.get('event')=='search' for e in t['events']) for t in traces if t['status']!='error']
    all_usage=[u for t in traces for u in t.get('usage',[])]
    all_usage += [u for t in traces for attempt in t.get('failed_attempts',[]) for u in attempt.get('usage',[])]
    by_provider={}
    for provider in sorted({t.get('provider') for t in traces if t.get('provider')}):
        provider_traces=[t for t in traces if t.get('provider')==provider]
        if not provider_traces: continue
        ids={t['query_id'] for t in provider_traces}
        provider_cases=[c for c in cases if c['id'] in ids]
        provider_reviews=[r for r in reviews if r['query_id'] in ids]
        by_provider[provider]={'models':sorted({t['model'] for t in provider_traces if t.get('model')}),
                               **aggregate(provider_cases,provider_traces,provider_reviews)}
    summary.update({'split':args.split,'configuration':args.config,'search_call_counts':{str(n):counts.count(n) for n in sorted(set(counts))},
                    'mean_elapsed_seconds':statistics.mean(t['elapsed_seconds'] for t in traces) if traces else None,
                    'input_tokens':sum(u['usage'].get('promptTokenCount',0) for u in all_usage),
                    'output_tokens':sum(u['usage'].get('candidatesTokenCount',0) for u in all_usage),
                    'thinking_tokens':sum(u['usage'].get('thoughtsTokenCount',0) for u in all_usage),
                    'total_tokens_reported':sum(u['usage'].get('totalTokenCount',0) for u in all_usage),
                    'failed_attempts':sum(len(t.get('failed_attempts',[])) for t in traces), 'by_provider':by_provider,
                    'usage_note':'API token counts are reported by the selected controller; dollar cost is not inferred from token counts.',
                    'cost':'Not inferred from tokens; consult account billing.'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
