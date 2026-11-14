"""Create Day 7 questions without reusing Day 6's observed test set."""
import json
import argparse
from pathlib import Path
from day6_retrieval import DAY6

ROOT = Path(__file__).resolve().parents[1]
MDPI = 'mdpi-entropy-2023-25-213'


def case(id_, split, question, answer, page, evidence):
    return {'id': id_, 'split': split, 'question': question, 'answerable': True,
            'expected_answer': answer, 'gold': [{'source_id': MDPI, 'page': page, 'evidence': evidence}]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6-root',type=Path,default=DAY6)
    args=parser.parse_args()
    source = json.loads((args.day6_root / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
    rows = [{'id': q['id'], 'split': 'dev', 'question': q['question'], 'answerable': True,
             'expected_answer': q['answer'], 'gold': [{'source_id': q['gold_source_id'],
             'page': q['gold_page'], 'evidence': q['evidence_quote']}]} for q in source if q['split'] == 'dev']
    for q in rows:
        if q['id'] == 'q03': q['expected_answer'] = 'Quantum-mechanical superposition.'
        if q['id'] == 'q04': q['question'] = 'What derivation is the first main aim of the paper?'
        if q['id'] == 'q07':
            q.update(case('q07', 'dev', 'Which two normalized initial states are mentioned as examples?',
                          'The coherent state and the squeezed state.', 2,
                          'such as the coherent state [11] or the squeezed state'))
        if q['id'] == 'q15':
            q.update(case('q15', 'dev', 'What is the oscillator initial state before its potential is removed?',
                          'Thermal equilibrium at temperature T.', 10,
                          'oscillator initially prepared in the state of thermal equilibrium at\ntemperature T'))
        if q['id'] == 'q10':
            q.update(case('q10', 'dev', 'Which operator satisfying evolution Equation (59) is an integral of motion?',
                          'The density operator.', 9,
                          'The density operator ˆρ(t) satisfying the evolution Equation (59) is the integral of motion.'))
    rows += [
        case('t01','test','Besides being an experimental tool, what can the optical tomogram describe?','Quantum states.',4,'the optical tomogram can also be employed to describe quantum\nstates'),
        case('t02','test','What physical observables do the operators q and p represent in the evolution discussion?','Position and momentum.',9,'where ˆq and ˆp are the position and momentum operators'),
        case('t03','test','What property does the evolution operator have when the Hamiltonian is Hermitian?','It is unitary.',9,'the evolution operator\nˆu(t) is the unitary operator'),
        case('t04','test','Which entropy is introduced for the conditional probability distributions?','Shannon entropy.',12,'Shannon entropy'),
        case('t05','test','How is the entropy S decomposed in the tomographic probability discussion?','As the sum S = S1 + S2.',12,'The entropy has the form of sum S = S1 + S2'),
    ]
    for id_, split, question, reason in [
        ('du01','dev','What measured experimental uncertainty in kelvin was reported for the oscillator temperature?', 'The indexed articles discuss theoretical states; no measured temperature uncertainty is specified.'),
        ('du02','dev','How many human participants were enrolled to validate the path-integral derivation?', 'The derivation is theoretical and the indexed corpus contains no participant count.'),
        ('tu01','test','Which instrument was used to measure the electric field in the charged-particle experiment?', 'The corpus presents theoretical evolution and specifies no such measurement instrument.'),
        ('tu02','test','What was the measured numerical value of the oscillator temperature in the laboratory experiment?', 'The indexed treatment uses the symbolic parameter T and supplies no laboratory measurement.')]:
        rows.append({'id':id_,'split':split,'question':question,'answerable':False,
                     'expected_behavior':'search_and_abstain','absence_reason':reason})
    path = ROOT / 'data' / 'benchmark_day7.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(rows)} questions to {path}')


if __name__ == '__main__': main()
