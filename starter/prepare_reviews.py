"""Prepare explicit review rows and a readable audit sheet for agent traces."""
import argparse
import json
from pathlib import Path
from retrieval_benchmark import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev','test'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--benchmark', type=Path, default=ROOT/'data'/'benchmark_day7.json')
    p.add_argument('--traces', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    traces_path = args.traces or ROOT/'starter'/'results'/'agent'/f'{args.split}_traces.json'
    cases = {q['id']: q for q in json.loads(args.benchmark.read_text(encoding='utf-8')) if q['split']==args.split}
    traces = [t for t in json.loads(traces_path.read_text(encoding='utf-8')) if t['configuration']==args.config and t['query_id'] in cases]
    rows = []
    for trace in traces:
        q = cases[trace['query_id']]
        if trace['status']=='error': continue
        row = {'query_id':q['id'],'context_grade':None,'citation_support':[None]*len(trace['citations']),
               'reason':'','citation_reasons':['']*len(trace['citations'])}
        if q['answerable']: row['answer_grade'] = None
        else: row['abstention_correct'] = None
        rows.append(row)
    if args.output.exists(): raise FileExistsError('Review file already exists; preserve existing labels')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Prepared {len(rows)} blank review rows at {args.output}; inspect actual traces and source before assigning labels.')


if __name__ == '__main__': main()
