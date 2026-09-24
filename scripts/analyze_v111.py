"""Offline, manifest-complete v111 analysis; semantic adjudication is separate."""
import argparse
import json
import math
import random
from pathlib import Path

from mini_agent_rl.hybrid_v110 import answer_metrics

ARMS = ('U', 'P', 'R', 'D')
CANDIDATES = ('P', 'R', 'D')
METRICS = ('strict_em', 'normalized_em', 'f1')


def bootstrap(values, resamples=10000):
    if resamples < 1:
        raise ValueError('resamples_must_be_positive')
    if not values:
        return {'mean': None, 'ci95': None, 'tasks': 0}
    rng = random.Random(42)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(resamples))
    return {'mean': sum(values) / n,
            'ci95': [means[min(int(resamples * .025), resamples - 1)],
                     means[min(int(resamples * .975), resamples - 1)]], 'tasks': n}


def analyze(inputs, rows, resamples=10000, expected_tasks=100, evaluated_arms=None):
    arms = tuple(ARMS if evaluated_arms is None else evaluated_arms)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError('invalid_evaluated_arms')
    candidates = tuple(a for a in CANDIDATES if a in arms)
    if not inputs:
        raise ValueError('empty_manifest')
    ids = [r['task_id'] for r in inputs]
    row_ids = [r['task_id'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(row_ids)) != len(row_ids) or set(row_ids) - set(ids):
        raise ValueError('manifest_result_identity_conflict')
    by = {r['task_id']: r for r in rows}
    details = []
    missing_answers = {arm: 0 for arm in arms}
    for source in inputs:
        row = by.get(source['task_id'], {})
        item = {'task_id': source['task_id'], 'gold': source['gold'],
                'original_answer': source['original_answer'],
                'A': answer_metrics(source['original_answer'], source['gold']),
                'status': row.get('status', 'missing'),
                'retrieval_queries': len(row.get('retrieval', []))}
        for arm in arms:
            answer = row.get('answers', {}).get(arm)
            if not isinstance(answer, str):
                missing_answers[arm] += 1
                answer = source['original_answer']
            item[arm + '_answer'] = answer
            item[arm] = answer_metrics(answer, source['gold'])
        matched = row.get('answers', {}).get('M')
        if isinstance(matched, str):
            item['M_answer'] = matched
            item['M'] = answer_metrics(matched, source['gold'])
        details.append(item)
    n = len(inputs)
    correct = [d for d in details if d['A']['normalized_em']]
    metrics = {arm: {m: sum(d[arm][m] for d in details) / n for m in METRICS}
               for arm in ('A',) + arms}
    preservation = {arm: (sum(d[arm]['normalized_em'] for d in correct) / len(correct)
                          if correct else None) for arm in arms}
    protocol = {arm: sum(by.get(i, {}).get('protocol', {}).get(arm, {}).get('valid') is True
                         for i in ids) / n for arm in arms}
    first_pass = {arm: sum(by.get(i, {}).get('protocol', {}).get(arm, {}).get('first_pass_valid') is True
                           for i in ids) / n for arm in arms}
    errors = sum(r.get('status') != 'ok' for r in rows)
    completed = sum(r.get('status') == 'ok' for r in rows)
    complete = n == expected_tasks and len(rows) == n and errors == 0 and not any(missing_answers.values())
    gates = {arm: bool(complete and metrics[arm]['normalized_em'] >= metrics['A']['normalized_em']
                       and metrics[arm]['f1'] >= metrics['A']['f1']
                       and preservation[arm] is not None and preservation[arm] >= .95
                       and protocol[arm] >= .98) for arm in candidates}
    calls = {}
    for arm in arms:
        counts = [by.get(i, {}).get('calls_by_arm', {}).get(arm) for i in ids]
        calls[arm] = (sum(counts) / n if all(isinstance(c, (int, float)) and not isinstance(c, bool)
                                            and math.isfinite(c) and c >= 0 for c in counts) else None)
    eligible = [a for a in candidates if gates[a]]
    # Compare call counts only when known for every tied candidate; missing is not expensive.
    rank = []
    while eligible:
        best = max((metrics[a]['normalized_em'], metrics[a]['f1'], preservation[a]) for a in eligible)
        tied = [a for a in eligible if (metrics[a]['normalized_em'], metrics[a]['f1'], preservation[a]) == best]
        if all(calls[a] is not None for a in tied):
            tied.sort(key=lambda a: (calls[a], ('D', 'P', 'R').index(a)))
        else:
            tied.sort(key=lambda a: ('D', 'P', 'R').index(a))
        rank.extend(tied)
        eligible = [a for a in eligible if a not in tied]

    def contrast(group, left, right):
        return {m: bootstrap([d[left][m] - d[right][m] for d in group], resamples) for m in METRICS}

    matched = [d for d in details if 'M' in d and d['retrieval_queries'] > 0]
    triggered = sum(d['retrieval_queries'] > 0 for d in details)
    report = {'tasks': n, 'expected_tasks': expected_tasks, 'evaluated_arms': list(arms), 'completed': completed,
              'missing_tasks': n - len(rows), 'missing_answers': missing_answers,
              'infrastructure_errors': errors, 'manifest_complete': complete,
              'metrics': metrics, 'correct_original_preserve': preservation,
              'baseline_correct_tasks': len(correct), 'protocol_valid_rate': protocol,
              'protocol_first_pass_valid_rate': first_pass, 'average_calls': calls,
              'numerical_gate': gates, 'eligible_ranking': rank,
              'selected_numerical_candidate': rank[0] if rank else None,
              'ranking_rule': 'EM, F1, preservation; calls only if known for every tied arm; otherwise D,P,R stable preference, not a cost claim',
              'semantic_review_status': 'pending', 'semantic_review_required_before_adoption': True,
              'adoption_approved': False,
              'paired_vs_A': {a: contrast(details, a, 'A') for a in arms},
              'transitions': {a: {'wrong_to_right': sum(not d['A']['normalized_em'] and bool(d[a]['normalized_em']) for d in details),
                                  'right_to_wrong': sum(bool(d['A']['normalized_em']) and not d[a]['normalized_em'] for d in details)} for a in arms},
              'matched_regeneration': {'retrieval_tasks': triggered, 'observed_tasks': len(matched),
                                       'complete_trigger_cohort': len(matched) == triggered,
                                       'conditional_only': True},
              'bootstrap': {'seed': 42, 'resamples': resamples, 'unit': 'question'}}
    if 'D' in arms:
        report['paired_vs_D'] = {a: contrast(details, a, 'D') for a in arms if a != 'D'}
    if {'P', 'U'} <= set(arms):
        report['paired_P_minus_U'] = contrast(details, 'P', 'U')
    if {'R', 'P'} <= set(arms):
        report['paired_R_minus_P'] = contrast(details, 'R', 'P')
    if 'R' in arms:
        report['matched_regeneration']['R_minus_M'] = contrast(matched, 'R', 'M')
    if 'P' in arms:
        report['matched_regeneration']['M_minus_P'] = contrast(matched, 'M', 'P')
    return report, details


def summarize(inputs, rows, resamples=10000, expected_tasks=100, evaluated_arms=None):
    return analyze(inputs, rows, resamples, expected_tasks, evaluated_arms)[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'results', 'output', 'details'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--arms', default=','.join(ARMS), help='Comma-separated evaluated arms, e.g. P,D')
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    report, details = analyze(read(args.inputs), read(args.results), evaluated_arms=args.arms.split(','))
    for path in (args.output, args.details):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    Path(args.details).write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in details), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
