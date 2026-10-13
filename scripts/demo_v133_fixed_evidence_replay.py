"""Replay frozen v118/Goal J outputs. This does not run a model."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J_MIRROR = ROOT / 'runs/v133-goal-j-transfer/autodl-results'
J_REMOTE = ROOT / 'runs/v133-goal-j-transfer'
J = J_MIRROR if (J_MIRROR / 'trajectories.jsonl').exists() else J_REMOTE
G = ROOT / 'runs/v133-goal-g-cpu/evaluation-frozen.jsonl'
N = ROOT / 'runs/v133-goal-n-v118-grpo/evaluation.jsonl'


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'new_eval', 'goal_G_regression'], default='new_eval')
    parser.add_argument('--case-id', default=None)
    args = parser.parse_args()
    data = {'train': J / 'train-frozen.jsonl', 'new_eval': J / 'evaluation-frozen.jsonl',
            'goal_G_regression': G}[args.split]
    candidates = read(data)
    row = next((x for x in candidates if x['case_id'] == args.case_id), None) if args.case_id else candidates[0]
    if row is None:
        parser.error('case ID is not in the selected frozen split')
    traces = [x for x in read(J / 'trajectories.jsonl') if x['split'] == args.split and x['case_id'] == row['case_id']]
    assert len(traces) == 2 and {x['arm'] for x in traces} == {'goal_J_sft', 'v118_new_eval' if args.split == 'new_eval' else 'v118'}
    if N.exists():
        rl = [x for x in read(N) if x['arm'] == 'goal_N_grpo' and x['split'] == args.split and x['case_id'] == row['case_id']]
        assert len(rl) == 1
        traces.append({'arm': 'goal_N_grpo', 'raw_output': rl[0]['raw'], 'em': rl[0]['em'], 'f1': rl[0]['f1']})
    output = {'mode': 'frozen_replay_no_generation', 'scope': 'synthetic two-search history; fixed visible evidence',
              'split': args.split, 'case_id': row['case_id'], 'question': row['messages'][1]['content'],
              'evidence': row['evidence_documents'], 'gold': row['official_answer'],
              'outputs': [{'adapter': x['arm'], 'action': x['raw_output'], 'strict_em': x['em'], 'f1': x['f1']}
                          for x in sorted(traces, key=lambda x: x['arm'])]}
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
