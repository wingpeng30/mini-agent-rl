"""Offline paired decomposition of the frozen v109 substring guard."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from mini_agent_rl.hybrid_answer import conservative_boundary_guard
from mini_agent_rl.hybrid_v110 import answer_metrics, safe_selection
from audit_v110_legacy import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('backups/v110-legacy-dev-inputs.jsonl'))
    p.add_argument('--results', type=Path, nargs='+', default=[Path('backups/v109') / name for name in
        ('dev20-results.jsonl', 'dev20-results-part2.jsonl', 'dev100-results-part3.jsonl',
         'dev100-results-part4.jsonl', 'dev100-results-part5.jsonl')])
    p.add_argument('--retries', type=Path, nargs='*', default=[Path('backups/v109/candidate-retries.jsonl')])
    p.add_argument('--output', type=Path, default=Path('reports/v110-legacy-guard-analysis.json'))
    p.add_argument('--markdown', type=Path, default=Path('reports/v110-legacy-guard-analysis.md'))
    args = p.parse_args()
    inputs, base, retry = load([args.inputs]), load(args.results), load(args.retries)
    ids = [x['record']['task_id'] for x in inputs]
    if len(ids) != len(set(ids)) or len(ids) != 100:
        raise ValueError('Expected exactly 100 unique verified development tasks')
    groups = [{}, {}]
    for attempts, grouped in zip((base, retry), groups):
        for attempt in attempts:
            task_id = attempt['record']['task_id']
            if task_id not in ids:
                raise ValueError(f'Unexpected task {task_id}')
            grouped.setdefault(task_id, []).append(attempt)
    rows = []
    for item in inputs:
        source = item['record']; task_id = source['task_id']
        if not source.get('provenance_valid'):
            raise ValueError(f'Unverified provenance: {task_id}')
        successes = [[x for x in grouped.get(task_id, []) if x['record'].get('status') == 'ok'] for grouped in groups]
        if len(successes[0]) != 1 or len(successes[1]) > 1:
            raise ValueError(f'Missing or conflicting successes: {task_id}')
        original = successes[0][0]['record']
        if original['gold'] != source['gold'] or original['original_answer'] != source['original_answer']:
            raise ValueError(f'Identity mismatch: {task_id}')
        chosen = successes[1][0] if successes[1] else successes[0][0]
        result = chosen['record'] if successes[1] else chosen['record']['e0']
        raw = result['selected'] if successes[1] else result['selected_answer']
        guarded = result['guarded_answer']
        reproduced, status = conservative_boundary_guard(raw, source['original_answer'], result['candidates'])
        if reproduced != guarded or status != result['guard_status']:
            raise ValueError(f'Cannot reproduce original guard: {task_id}')
        safe, safe_status = safe_selection(raw, source['original_answer'])
        answers = {'baseline': source['original_answer'], 'raw_selected': raw,
                   'legacy_guarded': guarded, 'v110_equality_only': safe}
        metrics = {name: answer_metrics(answer, source['gold']) for name, answer in answers.items()}
        rows.append({'task_id': task_id, 'question': source['question'], 'gold': source['gold'],
                     'answers': answers, 'metrics': metrics, 'guard_status': status,
                     'selection_status': result['selection_status'], 'safe_status': safe_status,
                     'retry_applied': bool(successes[1]), 'chosen_source': {k: v for k, v in chosen.items() if k != 'record'},
                     'guard_changed_answer': raw != guarded,
                     'guard_delta': {key: metrics['legacy_guarded'][key] - metrics['raw_selected'][key]
                                     for key in metrics['baseline']}})
    means = {name: {key: sum(r['metrics'][name][key] for r in rows) / 100
                    for key in rows[0]['metrics'][name]} for name in rows[0]['metrics']}
    transitions = {key: {'guard_improvements': [r['task_id'] for r in rows if r['guard_delta'][key] > 0],
                         'guard_regressions': [r['task_id'] for r in rows if r['guard_delta'][key] < 0],
                         'net_guard_gain': means['legacy_guarded'][key] - means['raw_selected'][key]}
                   for key in means['baseline']}
    report = {'version': 'v110', 'kind': 'offline_legacy_guard_ablation_not_new_experiment',
              'denominator': 100, 'coverage_complete': True, 'metrics': means, 'transitions': transitions,
              'guard_changed_answers': sum(r['guard_changed_answer'] for r in rows),
              'guard_reproduction_exact': 100, 'attempt_count': len(base), 'retry_attempt_count': len(retry),
              'failed_attempts': [x for x in base if x['record'].get('status') != 'ok'],
              'failed_retries': [x for x in retry if x['record'].get('status') != 'ok'],
              'retry_successes_applied': sum(r['retry_applied'] for r in rows),
              'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [args.inputs, *args.results, *args.retries]},
              'scorer': 'mini_agent_rl.hybrid_v110.answer_metrics',
              'limitations': ['Frozen v109 candidates and selections; this isolates only deterministic guard changes.',
                             'raw_selected includes legacy selector fallbacks; it is not unvalidated model output.',
                             'v110 equality-only applies safe_selection to legacy selected answers; it does not rerun citation validation.',
                             'strict_em is whitespace-trimmed exact match.'],
              'api_calls': 0, 'final_test_accessed': False, 'rows': rows}
    lines = ['# Frozen v109 guard decomposition', '',
             'All 100 verified legacy development tasks are retained. Successful explicit retries override base records; '
             'three failed shard attempts and two failed retries remain accounted for. All 100 saved guard outputs were reproduced.', '',
             '| Answer stage | Trimmed strict EM | Official normalized EM | Official F1 |',
             '|---|---:|---:|---:|']
    for name, values in means.items():
        lines.append(f"| {name} | {values['strict_em']:.4f} | {values['normalized_em']:.4f} | {values['f1']:.6f} |")
    lines += ['', f"The substring guard changed {report['guard_changed_answers']} answers.", '']
    for key, values in transitions.items():
        lines.append(f"- {key}: {len(values['guard_improvements'])} improvements, {len(values['guard_regressions'])} regressions; guard minus raw = {values['net_guard_gain']:+.6f}.")
    lines += ['', '## Every guard-changed answer', '',
              '| Task | Gold | Baseline | Raw selected | Guarded | Guard delta (strict / normalized / F1) |',
              '|---|---|---|---|---|---|']
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for r in rows:
        if r['guard_changed_answer']:
            values = [r['task_id'], r['gold'], r['answers']['baseline'], r['answers']['raw_selected'],
                      r['answers']['legacy_guarded'], ' / '.join(f'{v:+.4f}' for v in r['guard_delta'].values())]
            lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
    lines += ['', '## Interpretation limits', '', *['- ' + value for value in report['limitations']], '',
              'No new model calls were made and no final-test data were accessed. Full task-level scores, provenance hashes, failures, and retry precedence are in the JSON report.', '']
    for path in (args.output, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.markdown.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('denominator', 'metrics', 'guard_changed_answers', 'transitions')}, indent=2))


if __name__ == '__main__':
    main()
