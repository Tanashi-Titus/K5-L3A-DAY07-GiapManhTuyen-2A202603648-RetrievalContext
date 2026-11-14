"""Display a case, final context and cited excerpts for manual audit only."""
import argparse
import json
from retrieval_benchmark import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--query',required=True)
    args=p.parse_args()
    cases=json.loads((ROOT/'data'/'benchmark_day7.json').read_text(encoding='utf8'))
    traces=json.loads((ROOT/'starter'/'results'/'agent'/f'{args.split}_traces.json').read_text(encoding='utf8'))
    case=next(c for c in cases if c['split']==args.split and c['id']==args.query)
    trace=next(t for t in traces if t['configuration']==args.config and t['query_id']==args.query)
    searches=[e for e in trace['events'] if e.get('event')=='search']
    chunks={c['id']:c for e in searches for c in e['results']}
    print('QUESTION:',case['question'])
    print('REFERENCE:',case.get('expected_answer',case.get('absence_reason')))
    print('STATUS:',trace['status'],'PROVIDER:',trace.get('provider','gemini'),'MODEL:',trace.get('model'))
    print('ANSWER:',trace.get('answer',''))
    print('CONTEXT:\n',searches[-1]['context'] if searches else '(no successful search)')
    for id_ in trace['citations']:
        c=chunks[id_]
        print(f"\nCITATION [{id_}] {c['source_id']} page {c['page']}:\n{c['text']}")
    if trace.get('controller_decisions'):
        print('\nFINAL MODEL DECISION:',json.dumps(trace['controller_decisions'][-1],ensure_ascii=False))


if __name__=='__main__': main()
