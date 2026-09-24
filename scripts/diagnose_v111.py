"""Offline train-only diagnosis. Never imports or calls the paid runner."""
import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from analyze_v111 import analyze
from mini_agent_rl.hybrid_v110 import answer_metrics


def _calls(row):
    for index, call in enumerate(row.get('initial_calls', [])):
        yield 'initial:' + str(index), call
    for field in ('retrieval_rounds', 'matched_rounds'):
        for index, round_ in enumerate(row.get(field, [])):
            for subindex, call in enumerate(round_.get('calls', [])):
                yield f'{field}:{index + 1}:{subindex}', call
    if row.get('direct_call'):
        yield 'direct', row['direct_call']


def ledger_summary(path, stage='v111-train'):
    """Aggregate one explicit phase using a read-only SQLite connection."""
    if stage != 'v111-train':
        raise ValueError('train_phase_only')
    phases = {}
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        con.row_factory = sqlite3.Row
        for row in con.execute('SELECT stage,config,status,charged,usage,raw_response FROM calls WHERE stage=?', (stage,)):
            cfg = json.loads(row['config'])
            kind = cfg.get('kind', 'unknown')
            # Remove only the trailing implementation fingerprint, retain mechanism/round.
            kind = re.sub(r':[0-9a-f]{64}$', '', kind)
            entry = phases.setdefault(kind, {'calls': 0, 'charged_upper_cny': 0.0,
                'status_histogram': Counter(), 'served_models': Counter(), 'requested_models': Counter(),
                'prompt_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0,
                'reasoning_usage_missing': 0, 'latency_seconds': 0.0,
                'completion_token_histogram': Counter(), 'reasoning_token_histogram': Counter()})
            usage = json.loads(row['usage']) if row['usage'] else {}
            try:
                raw = json.loads(row['raw_response']) if row['raw_response'] else {}
            except (ValueError, TypeError):
                raw = {}
            if not isinstance(raw, dict):
                raw = {}
            reasoning = usage.get('completion_tokens_details', {}).get('reasoning_tokens', usage.get('reasoning_tokens'))
            entry['calls'] += 1
            entry['charged_upper_cny'] += row['charged']
            entry['status_histogram'][row['status']] += 1
            entry['served_models'][raw.get('model', 'unavailable')] += 1
            entry['requested_models'][cfg.get('request', {}).get('model', cfg.get('model', 'unavailable'))] += 1
            entry['prompt_tokens'] += usage.get('prompt_tokens', 0)
            entry['completion_tokens'] += usage.get('completion_tokens', 0)
            entry['latency_seconds'] += usage.get('latency_seconds', 0)
            entry['completion_token_histogram'][str(usage.get('completion_tokens', 'unavailable'))] += 1
            entry['reasoning_token_histogram'][str(reasoning) if reasoning is not None else 'unavailable'] += 1
            if reasoning is None:
                entry['reasoning_usage_missing'] += 1
            else:
                entry['reasoning_tokens'] += reasoning
    finally:
        con.close()
    return {'stage': stage, 'by_kind': phases,
            'calls': sum(p['calls'] for p in phases.values()),
            'charged_upper_cny': sum(p['charged_upper_cny'] for p in phases.values()),
            'cost_is_conservative_ledger_accounting_not_billing': True,
            'shared_upstream_calls_counted_once': True}


def diagnose(inputs, rows, resamples=10000, expected_tasks=100):
    analysis, paired = analyze(inputs, rows, resamples=resamples, expected_tasks=expected_tasks)
    if not analysis['manifest_complete']:
        raise ValueError('complete_train100_required_before_diagnosis')
    by = {r['task_id']: r for r in rows}
    details = []
    categories = Counter()
    original_status = Counter()
    failures = Counter()
    error_kinds = Counter()
    attempt_counts = Counter()
    for source, scores in zip(inputs, paired):
        row = by[source['task_id']]
        base_correct = bool(scores['A']['normalized_em'])
        covered = any(answer_metrics(c['answer'], source['gold'])['normalized_em']
                      for c in row.get('initial_candidates', []) if isinstance(c.get('answer'), str))
        category = ('baseline_correct' if base_correct else 'candidate_missing' if not covered else
                    'candidate_recovered_by_U' if scores['U']['normalized_em'] else 'candidate_available_U_wrong')
        categories[category] += 1
        first = row.get('reviews', [{}])[0] if row.get('reviews') else {}
        original = first.get('assessments', {}).get('original', {})
        supported = original.get('valid') is True and original.get('status') == 'supported'
        support_reviews = [index for index, review in enumerate(row.get('reviews', []))
                           if review.get('assessments', {}).get('original', {}).get('valid') is True
                           and review['assessments']['original'].get('status') == 'supported']
        original_status[f"{original.get('status', 'missing')}:valid={original.get('valid') is True}:Acorrect={base_correct}"] += 1
        attempts = []
        for phase, call in _calls(row):
            attempt_counts[str(len(call.get('attempts', [])))] += 1
            for attempt in call.get('attempts', []):
                if attempt.get('valid') is True:
                    continue
                errors = attempt.get('errors', [])
                failures.update(errors)
                error_kinds[attempt.get('error_kind', 'unspecified')] += 1
                attempts.append({'phase': phase, 'attempt': attempt.get('attempt'),
                                 'error_kind': attempt.get('error_kind'), 'errors': errors,
                                 'eventual_call_valid': call.get('valid')})
        item = {**scores, 'category': category, 'candidate_contains_gold': covered,
                'first_original_assessment': original,
                'first_supported_but_official_em_wrong': supported and not base_correct,
                'supported_but_official_em_wrong': bool(support_reviews) and not base_correct,
                'original_supported_review_indices': support_reviews,
                'protection_changed': scores['P_answer'] != scores['U_answer'],
                'protection_rescued_original_correct': base_correct and not scores['U']['normalized_em'] and bool(scores['P']['normalized_em']),
                'protection_lost_repair': not base_correct and bool(scores['U']['normalized_em']) and not scores['P']['normalized_em'],
                'failed_attempts': attempts,
                'protocol': row.get('protocol', {}),
                'retrieval_rounds': len(row.get('retrieval_rounds', []))}
        details.append(item)
    wrong = [d for d in details if not d['A']['normalized_em']]
    supported_wrong = [d for d in details if d['supported_but_official_em_wrong']]
    report = {'version': 'v111', 'training_development_only': True, 'complete_train100': len(inputs) == 100,
              'tasks': len(inputs), 'analysis': analysis, 'error_categories': dict(categories),
              'candidate_recall_on_baseline_wrong': {'numerator': sum(d['candidate_contains_gold'] for d in wrong), 'denominator': len(wrong)},
              'original_assessment_cross_tab': dict(original_status),
              'supported_but_official_em_wrong': {'count': len(supported_wrong), 'task_ids': [d['task_id'] for d in supported_wrong],
                  'first_review_count': sum(d['first_supported_but_official_em_wrong'] for d in details),
                  'interpretation': 'Scoring disagreement is not proof of factual false support; all such cases require evidence review, including unchanged answers.'},
              'protection': {'changed': sum(d['protection_changed'] for d in details),
                  'rescued_original_correct': sum(d['protection_rescued_original_correct'] for d in details),
                  'lost_repairs': sum(d['protection_lost_repair'] for d in details)},
              'effects': {'P_minus_U': analysis['paired_P_minus_U'], 'R_minus_P': analysis['paired_R_minus_P'],
                         'D_minus_A': analysis['paired_vs_A']['D']},
              'retrieval_conditional': analysis['matched_regeneration'],
              'protocol': {'failed_attempt_causes': dict(failures), 'error_kinds': dict(error_kinds),
                           'attempts_per_call_histogram': dict(attempt_counts),
                           'failed_attempts': sum(len(d['failed_attempts']) for d in details),
                           'note': 'Each physical call group counted once, not once per shared arm; a failed attempt may have multiple causes.'},
              'caveats': ['Development data do not establish independent generalization.',
                          'Same-model support assessments and task-agent evidence review are not independent human truth.',
                          'R-minus-P includes retrieval plus regeneration; conditional R-minus-M controls extra regeneration only on the fixed trigger cohort.',
                          'Semantic review remains pending; diagnosis does not authorize adoption.']}
    return report, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='backups/v110-train-inputs.jsonl')
    parser.add_argument('--results', default='backups/v111-train-results.jsonl')
    parser.add_argument('--ledger', default='runs/v110-api.sqlite')
    parser.add_argument('--output', default='reports/v111-diagnosis.json')
    parser.add_argument('--details', default='reports/v111-diagnosis-details.jsonl')
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    report, details = diagnose(read(args.inputs), read(args.results))
    report['ledger'] = ledger_summary(args.ledger)
    for name in (args.output, args.details):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    Path(args.details).write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in details), encoding='utf-8')
    print(json.dumps({'tasks': report['tasks'], 'output': args.output, 'details': args.details}))


if __name__ == '__main__':
    main()
