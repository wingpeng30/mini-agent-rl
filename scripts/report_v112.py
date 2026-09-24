"""Build an evidence-bound v112 report and a new, verified archive (no API calls).

Required remote receipt: schema_version=1, version='v112', captured_at (UTC
ISO-8601), instance_running=true, gpu={name, memory_used_mib},
source_adapter_sha256, remote_same_path_ledger_present=false,
independent100_evaluated=false, final_test_accessed=false, access_claim_basis,
and
artifacts=[{path: repository-relative local receipt/log path, sha256: hex}].
The artifact list must bind training, remote replay, comparison, and all three
raw evaluation JSON files supplied on the command line. Collect this receipt
only after those remote outputs have settled. No protected evaluation input is
opened by this script. A numerical failure is a valid result to archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tarfile
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def local(name):
    p = ROOT / name
    resolved = p.resolve()
    if not resolved.is_relative_to(ROOT) or p.is_symlink():
        raise ValueError('unsafe_artifact_path:' + str(name))
    lower = str(name).lower().replace('_', '').replace('-', '')
    if any(term in lower for term in ('finaltest', 'independent100', 'heldout100')):
        raise ValueError('protected_artifact_path:' + str(name))
    return resolved


def read(name):
    return json.loads(local(name).read_text(encoding='utf-8'))


def junit(path):
    tree = ET.parse(path)
    cases = list(tree.iter('testcase'))
    if not cases:
        raise ValueError('junit_no_testcases')
    result = {'tests': len(cases), 'failures': sum(c.find('failure') is not None for c in cases),
              'errors': sum(c.find('error') is not None for c in cases),
              'skipped': sum(c.find('skipped') is not None for c in cases)}
    result['passed'] = result['tests'] - result['failures'] - result['errors'] - result['skipped']
    if result['failures'] or result['errors']:
        raise ValueError('junit_regressions_failed')
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base-tasks', 'candidate-tasks', 'candidate-states'):
        suffix = 'baseline-tasks' if arg == 'base-tasks' else arg
        parser.add_argument('--' + arg, default=f'reports/v112-replay2-{suffix}.json',
                            help='settled internal-only raw evaluation JSON')
    parser.add_argument('--comparison', default='reports/v112-internal-comparison.json')
    parser.add_argument('--remote-receipt', default='reports/v112-remote-receipt.json')
    args = parser.parse_args()
    raw_names = [args.base_tasks, args.candidate_tasks, args.candidate_states]
    required = ['reports/v112-summary.json', 'reports/v112-training-report.json',
                'reports/v112-prompt-replay-local.json', 'reports/v112-prompt-replay-remote.json',
                'reports/v112-tests.xml', 'data/evidence-sft-v112/manifest.json',
                args.comparison, args.remote_receipt, *raw_names]
    # Require all evidence before creating any output or archive.
    for name in required:
        if not local(name).is_file():
            raise FileNotFoundError(name)
    initial_hashes = {name: sha(local(name)) for name in required}
    summary = read(required[0]); training = read(required[1])
    comparison = read(args.comparison); remote = read(args.remote_receipt)
    manifest = read('data/evidence-sft-v112/manifest.json')
    tests = junit(local('reports/v112-tests.xml'))
    gate = summary['data_gate']
    if gate.get('passed') is not True or summary.get('budget_policy', {}).get('mode') != 'uncapped':
        raise ValueError('approved_data_and_uncapped_authorization_required')
    before, after = training['source_before'], training['source_after']
    source_sha = gate['source_adapter_sha256']
    formal = training['phases']['formal']
    history = formal['metrics']['train_metrics']
    steps = max(r.get('step', 0) for r in history)
    if (training.get('execute') is not True or before != after or not before or
        before.get('adapter_model.safetensors') != source_sha or
        formal.get('reload_ok') is not True or steps != training['formal_updates'] or steps <= 0):
        raise ValueError('training_completion_or_source_integrity_failed')
    for name in ('reports/v112-prompt-replay-local.json', 'reports/v112-prompt-replay-remote.json'):
        replay = read(name)
        if replay.get('passed') is not True:
            raise ValueError('prompt_replay_failed:' + name)
        for split in ('train', 'validation'):
            part = replay['splits'][split]; artifact = gate['artifacts'][split]
            if (part['failures'] or part['sha256'] != artifact['sha256'] or
                part['count'] != artifact['rows'] or sha(local(artifact['path'])) != artifact['sha256']):
                raise ValueError('prompt_replay_data_mismatch')
    expected_keys = {'schema_version', 'version', 'captured_at', 'instance_running', 'gpu',
                     'source_adapter_sha256', 'remote_same_path_ledger_present',
                     'independent100_evaluated', 'final_test_accessed', 'access_claim_basis', 'artifacts'}
    if set(remote) != expected_keys or remote['schema_version'] != 1 or remote['version'] != 'v112':
        raise ValueError('remote_receipt_schema')
    captured = datetime.fromisoformat(remote['captured_at'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if captured.utcoffset() is None or not 0 <= age <= 86400:
        raise ValueError('remote_receipt_stale_or_future')
    if (remote['instance_running'] is not True or remote['source_adapter_sha256'] != source_sha or
        any(remote[k] is not False for k in ('remote_same_path_ledger_present',
                                            'independent100_evaluated', 'final_test_accessed'))):
        raise ValueError('remote_receipt_safety_failed')
    gpu = remote['gpu']
    if (set(gpu) != {'name', 'memory_used_mib'} or not isinstance(gpu['name'], str) or not gpu['name'] or
        type(gpu['memory_used_mib']) is not int or gpu['memory_used_mib'] < 0):
        raise ValueError('gpu_receipt_schema')
    if not isinstance(remote['access_claim_basis'], str) or not remote['access_claim_basis'].strip():
        raise ValueError('access_claim_basis_required')
    bound = {}
    bound_files = set()
    adapter_mapping = {
        f'checkpoints/v112-evidence-sft-seed42/formal/{name}': f'backups/v112-formal-adapter/{name}'
        for name in ('adapter_model.safetensors', 'adapter_config.json')}
    for item in remote['artifacts']:
        if set(item) != {'path', 'sha256'} or item['path'] in bound:
            raise ValueError('remote_artifact_schema')
        disk = local(adapter_mapping.get(item['path'], item['path']))
        if sha(disk) != item['sha256']:
            raise ValueError('remote_artifact_hash_mismatch')
        bound[item['path']] = item['sha256']
        bound_files.add(disk)
    for name in [required[1], 'reports/v112-prompt-replay-remote.json', args.comparison, *raw_names]:
        if name not in bound:
            raise ValueError('remote_artifact_binding_missing:' + name)
    # Recompute comparison from raw internal logs; never accept a hand-edited pass.
    from evaluate_v112 import compare_reports
    raw = [read(n) for n in raw_names]
    for r in raw[:2]:
        if set(r['expected_ids']) != set(manifest['internal_ids']):
            raise ValueError('not_frozen_internal_questions')
    computed = compare_reports(*raw)
    for remote_name in adapter_mapping:
        if remote_name not in bound:
            raise ValueError('adapter_backup_receipt_missing:' + remote_name)
        filename = Path(remote_name).name
        if raw[1]['fingerprint']['adapter_files'].get(filename) != bound[remote_name]:
            raise ValueError('archived_adapter_not_evaluated_adapter')
    if comparison != computed:
        raise ValueError('comparison_not_reproduced_from_raw')
    numeric = bool(comparison['numeric_gates']) and all(v is True for v in comparison['numeric_gates'].values())
    # Semantic acceptance is intentionally not inferred from action/token metrics.
    eligible = False
    decision = ('数值门槛未通过：停止独立100题评测及GRPO。' if not numeric else
                '数值门槛通过，但语义审核仍待完成：不得宣称内部验收通过，停止独立100题评测及GRPO。')
    archive_dir = ROOT / 'backups' / ('v112-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    archive_dir.mkdir(parents=True, exist_ok=False)
    snapshot = archive_dir / 'v110-api.sqlite'
    ledger = local('runs/v110-api.sqlite')
    # SQLite backup is an internally consistent snapshot, even with a WAL source.
    with sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(snapshot) as dst:
        src.backup(dst)
    with sqlite3.connect(snapshot.as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('pragma integrity_check').fetchone()[0] != 'ok':
            raise ValueError('ledger_integrity_failed')
        rows = db.execute('select stage,status,count(*),sum(charged) from calls group by stage,status').fetchall()
        old = db.execute("select * from calls where stage not like 'v112-%' order by key").fetchall()
        if any(status != 'complete' for _, status, _, _ in rows):
            raise ValueError('unsettled_ledger_no_final_archive')
        if digest(old) != manifest['opening_ledger']['rows_sha256']:
            raise ValueError('historical_ledger_changed')
    if any(cost is None or not math.isfinite(cost) or cost < 0 for _, _, _, cost in rows):
        raise ValueError('invalid_ledger_cost')
    accounting = {'authority': 'local runs/v110-api.sqlite', 'groups': rows,
                  'calls': sum(r[2] for r in rows), 'new_calls': sum(r[2] for r in rows if r[0].startswith('v112-')),
                  'cumulative_cny': sum(r[3] for r in rows), 'new_cny': sum(r[3] for r in rows if r[0].startswith('v112-')),
                  'unresolved': 0, 'prior_rows_sha256': digest(old), 'policy': 'uncapped'}
    receipt = {'version': 'v112', 'generated_at': datetime.now(timezone.utc).isoformat(),
               'data_gate_passed': True, 'formal_updates_completed': steps, 'source_unchanged': True,
               'source_adapter_sha256': source_sha, 'tests': tests, 'numeric_passed': numeric,
               'semantic_audit': comparison['semantic_audit'], 'internal_eligible': eligible,
               'decision': decision, 'independent100_evaluated': False, 'final_test_accessed': False,
               'remote_receipt': remote, 'accounting': accounting, 'evidence_sha256': initial_hashes,
               'adapter_backup': adapter_mapping, 'access_claim_basis': remote['access_claim_basis']}
    counts = gate['counts']
    report = '# v112：证据约束 SFT 实际结果\n\n' + decision + '\n\n'
    report += (f"实际完成独立 adapter SFT {steps} 步并重新加载，源文件训练前后哈希一致。"
               '训练 loss 仅衡量优化目标，不是问答准确率。v111 的78%为 API 混合流程结果，不是本地模型成绩。\n\n')
    report += '| 数据 | 状态 | 问题 | 回答 | 搜索 |\n|---|---:|---:|---:|---:|\n'
    for split, values in counts.items():
        report += f"| {split} | {values['states']} | {values['questions']} | {values['actions'].get('answer',0)} | {values['actions'].get('search',0)} |\n"
    report += '\n双审结果不是人工真值。内部集来自训练来源中的固定隔离题，不宣称从未用于历史实验。\n\n'
    coverage=gate['review_coverage']
    report += (f"共双审{coverage['api']}个候选，隔离{coverage['quarantined']}个被否决、存在分歧或验证失败的状态；"
               f"另有{coverage['pending']}个未双审状态不用于训练。gold不进入审核输入，争议目标不直接转成训练标签。\n\n")
    report += '内部配对结果（固定50题，原始日志重算）：\n\n| 指标 | v101 | v112 |\n|---|---:|---:|\n'
    report += f"| 规范化 EM | {comparison['base_em']:.2%} | {comparison['candidate_em']:.2%} |\n"
    report += f"| Token F1 | {comparison['base_f1']:.2%} | {comparison['candidate_f1']:.2%} |\n\n"
    state=comparison['state_metrics']
    report += (f"错转对{comparison['gains']}题、对转错{comparison['losses']}题，净增{comparison['net_improvement']}题；"
               f"原正确答案保留{comparison['preserved']}/{comparison['base_correct']}（{comparison['preservation']:.2%}）。"
               f"113状态中，充分证据答案目标匹配率{state['supported_answer_correct_rate']:.2%}，"
               f"不足证据搜索动作率{state['search_action_rate']:.2%}，协议合法率{state['protocol_rate']:.2%}。\n\n")
    report += '未通过门槛：'+', '.join(k for k,v in comparison['numeric_gates'].items() if not v)+'。语义审核未完成，不将评分变化直接宣称为事实纠错。\n\n'
    baseline_state=read('reports/v112-replay2-baseline-states.json')['metrics']
    report += (f"行为诊断：不足证据状态的搜索动作率从{baseline_state['search_action_rate']:.2%}升至"
               f"{state['search_action_rate']:.2%}，但充分证据答案目标匹配率从"
               f"{baseline_state['supported_answer_correct_rate']:.2%}到{state['supported_answer_correct_rate']:.2%}，没有改善。"
               '这说明动作选择的改善尚未稳定转化为端到端正确率收益。下一轮应先在训练来源中审核答案保留与证据范围问题，'
               '不得为了本轮内部错误逐题改标签、反复调参后仍声称该内部集未使用。独立100题继续保留。\n\n')
    report += ('首次验收因单batch与flat token数组形状不同停止，0条计分。严格规范化后另启replay2，'
               '逐token一致性检查保持启用，原始失败记录保留。构建时长度元数据错误也已记录；'
               '全513条在本地和远端复核均无超限或输入变化（最大787/794 token），训练前置检查亦独立验证长度和mask。'
               '评测后修正构建器，未重建冻结数据，实际执行旧构建器另存备份。\n\n')
    report += f"JUnit 实测：{tests['passed']}通过、{tests['failures']}失败、{tests['errors']}错误、{tests['skipped']}跳过；本地及远端 prompt replay 均通过且绑定同一数据哈希。\n\n"
    report += f"API账本已结算：本次{accounting['new_calls']}次，保守记账费用{accounting['new_cny']:.6f}元；累计{accounting['calls']}次、{accounting['cumulative_cny']:.6f}元。预算已授权取消上限，不报告余额。\n\n"
    report += f"远端回执时间 {remote['captured_at']}：GPU {gpu['name']}，占用 {gpu['memory_used_mib']} MiB，实例保持运行。独立100题及final-test未访问的声明依据本轮代码执行范围，不是全系统文件访问审计；远端同名账本不存在。\n\n"
    report += '独立 adapter 权重及配置从远端下载到 backups/v112-formal-adapter，已与远端回执及实际评测模型哈希核对，纳入本次归档。\n'
    local('docs/v112-evidence-sft-results.md').write_text(report, encoding='utf-8')
    verification = local('reports/v112-final-verification.json')
    write_json(verification, receipt)
    files = {local(n) for n in required}
    files.update(bound_files)
    files.add(local('configs/v112-budget-policy.json'))
    files.add(local('backups/v112-executed-build_v112.py'))
    files.add(local('backups/v112-api-pretrain.sqlite'))
    files.update(local(str(p.relative_to(ROOT))) for p in (ROOT/'backups/v112-formal-adapter').rglob('*') if p.is_file())
    for directory, pattern in [('scripts','*v112*'), ('tests','*v112*'), ('reports','v112*'),
                               ('docs','v112*'), ('src/mini_agent_rl','*v112*')]:
        for entry in (ROOT / directory).glob(pattern):
            if entry.is_dir():
                files.update(local(str(p.relative_to(ROOT))) for p in entry.rglob('*') if p.is_file())
            elif entry.is_file():
                files.add(local(str(entry.relative_to(ROOT))))
    files.update(local(str(p.relative_to(ROOT))) for p in (ROOT/'data/evidence-sft-v112').rglob('*') if p.is_file())
    files.add(snapshot)
    # Raw evaluation sidecars (SQLite trajectories, logs) are included explicitly.
    for name in raw_names:
        p = local(name)
        files.update(local(str(x.relative_to(ROOT))) for x in p.parent.glob(p.name + '*') if x.is_file())
    for name, expected in initial_hashes.items():
        if sha(local(name)) != expected:
            raise ValueError('input_changed_during_report:' + name)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    inventory = archive_dir / 'manifest.json'
    write_json(inventory, hashes)
    archive = archive_dir / 'artifacts.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in sorted(hashes):
            tar.add(ROOT/name, arcname=name, recursive=False)
        tar.add(inventory, arcname='archive-manifest.json', recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        if set(tar.getnames()) != set(hashes) | {'archive-manifest.json'}:
            raise ValueError('archive_member_set_mismatch')
        for name, expected in hashes.items():
            stream = tar.extractfile(name)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError('archive_hash_mismatch:' + name)
        if tar.extractfile('archive-manifest.json').read() != inventory.read_bytes():
            raise ValueError('archive_manifest_mismatch')
    for name, expected in hashes.items():
        if sha(ROOT/name) != expected:
            raise ValueError('artifact_changed_during_archive:' + name)
    # Separate seal avoids a self-referential archive hash. Archived verification
    # binds all evidence; this sidecar records subsequent archive verification.
    seal = {'archive': archive.relative_to(ROOT).as_posix(), 'sha256': sha(archive),
            'bytes': archive.stat().st_size, 'payload_files_verified': len(hashes),
            'verification_sha256': hashes[verification.relative_to(ROOT).as_posix()]}
    write_json(archive_dir/'archive-verification.json', seal)
    print(json.dumps({'decision': decision, 'archive': seal}, ensure_ascii=False))


if __name__ == '__main__':
    main()
