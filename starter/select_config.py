"""Lock the dev choice before running the independent test split."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from agent_metrics import aggregate
from gemini_controller import GeminiController
from groq_controller import GroqController
from openai_controller import OpenAIController
from retrieval_benchmark import CONFIGS, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chosen', choices=CONFIGS, required=True)
    p.add_argument('--baseline', choices=CONFIGS, required=True)
    p.add_argument('--chosen-reviews', type=Path, required=True)
    p.add_argument('--baseline-reviews', type=Path, required=True)
    p.add_argument('--rationale', required=True)
    p.add_argument('--output-root', type=Path, default=ROOT/'starter'/'results')
    args = p.parse_args()
    if args.chosen == args.baseline: raise ValueError('Choose distinct baseline and configuration')
    results = args.output_root
    selection_path = results/'retrieval'/'selection.json'
    if selection_path.exists() or (results/'retrieval'/'test_traces.json').exists() or (results/'agent'/'test_traces.json').exists():
        raise FileExistsError('Selection or test results already exist; preserve the original evaluation')
    benchmark = ROOT/'data'/'benchmark_day7.json'
    cases=[c for c in json.loads(benchmark.read_text(encoding='utf8')) if c['split']=='dev']
    traces=json.loads((results/'agent'/'dev_traces.json').read_text(encoding='utf8'))
    provider_counts={}
    for name in CONFIGS:
        rows=[t for t in traces if t['configuration']==name and t.get('controller_version') in
              (GeminiController.version,GroqController.version,OpenAIController.version)]
        if len(rows)!=len(cases) or any(t['status']=='error' for t in rows):
            raise ValueError(f'Complete dev agent runs for {name} before selection')
        if any(t['question'] != next(c['question'] for c in cases if c['id']==t['query_id']) for t in rows):
            raise ValueError(f'Outdated question in dev traces for {name}')
        provider_counts[name]={provider:sum(t.get('provider')==provider for t in rows)
                               for provider in ('openai','gemini','groq') if any(t.get('provider')==provider for t in rows)}
    for name, path in ((args.chosen,args.chosen_reviews),(args.baseline,args.baseline_reviews)):
        rows=[t for t in traces if t['configuration']==name]
        aggregate(cases, rows, json.loads(path.read_text(encoding='utf8')))
    selection={'chosen':args.chosen,'baseline':args.baseline,'rationale':args.rationale,
               'benchmark_sha256':hashlib.sha256(benchmark.read_bytes()).hexdigest(),
               'controller_versions':sorted({t.get('controller_version') for t in traces}),
               'provider_counts':provider_counts,
               'recorded_utc':datetime.now(timezone.utc).isoformat(),
               'criteria':'report/dev_selection_criteria.md'}
    selection_path.parent.mkdir(parents=True,exist_ok=True)
    selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
