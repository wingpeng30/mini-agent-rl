"""Bounded CPU audit of the frozen Goal J training/evaluation records."""
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor
from mini_agent_rl.training.sft import encode_assistant_only

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'runs/v133-goal-j-transfer'
K = ROOT / 'runs/v133-goal-k-attribution/autodl-results'
G = ROOT / 'runs/v133-goal-g-cpu/evaluation-frozen.jsonl'
OUT = ROOT / 'runs/v133-goal-m-readiness'


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((J / 'freeze-manifest.json').read_text(encoding='utf-8'))
    sets = {'train': rows(J / 'train-frozen.jsonl'), 'evaluation': rows(J / 'evaluation-frozen.jsonl'),
            'goal_G_regression': rows(G)}
    expected = {'train': manifest['train_sha256'], 'evaluation': manifest['evaluation_sha256'],
                'goal_G_regression': manifest['input_hashes']['runs/v133-goal-g-cpu/evaluation-frozen.jsonl']}
    checks = {}
    for split, values in sets.items():
        path = {'train': J / 'train-frozen.jsonl', 'evaluation': J / 'evaluation-frozen.jsonl',
                'goal_G_regression': G}[split]
        assert digest(path) == expected[split], split
        assert len(values) == 16 and len({r['case_id'] for r in values}) == 16, split
        checks[split] = {'count': 16, 'sha256': digest(path)}
    assert not (set(r['case_id'] for r in sets['train']) & set(r['case_id'] for r in sets['evaluation']))
    dispositions = {r['source_id']: r for r in rows(J / 'review-disposition.jsonl')}
    for split in ('train', 'evaluation'):
        for row in sets[split]:
            review = dispositions[row['source_id']]
            assert review['split'] == split and review['disposition'] == 'qualified'
            assert review['deepseek_valid'] and review['deepseek_approved']
            assert review['luna_valid'] and review['luna_approved']
            assert row['messages'][row['target_assistant_index']]['role'] == 'assistant'
            assert json.loads(row['messages'][row['target_assistant_index']]['content']) == {'action': 'answer', 'answer': row['official_answer']}
            assert [m['role'] for m in row['messages']] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']
            for doc, msg in zip(row['evidence_documents'], [row['messages'][3], row['messages'][5]]):
                assert doc in json.loads(msg['content'])['result']
    proc = AutoProcessor.from_pretrained('/root/autodl-tmp/qwen_08b', local_files_only=True)
    token_rows = []
    for split, values in sets.items():
        for row in values:
            idx = row['target_assistant_index']
            prompt = row['messages'][:idx]
            ids = proc.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            ids = list(ids[0] if ids and isinstance(ids[0], list) else ids)
            encoded = encode_assistant_only(proc, row['messages'], max_length=2176, target_assistant_index=idx, thinking_enabled=False)
            labels = encoded['labels']; positions = [i for i, x in enumerate(labels) if x != -100]
            assert encoded['input_ids'][:len(ids)] == ids and all(x == -100 for x in labels[:len(ids)])
            assert len(ids) <= 2048 and len(encoded['input_ids']) <= 2176 and len(positions) <= 128
            assert positions == list(range(positions[0], positions[-1] + 1))
            assert positions[0] >= len(ids) and len(labels)-1-positions[-1] <= 1, (split, row['case_id'], len(ids), positions[0], positions[-1], len(labels))
            assert labels[positions[-1]] == proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
            if split in ('train', 'evaluation'):
                assert hashlib.sha256(b''.join(int(i).to_bytes(4, 'little') for i in ids)).hexdigest() == row['cpu_check']['prompt_sha256']
            token_rows.append({'split': split, 'case_id': row['case_id'], 'prompt_tokens': len(ids),
                               'target_tokens': len(positions), 'total_tokens': len(labels), 'status': 'pass'})
    (OUT / 'cpu-audit.json').write_text(json.dumps({'sets': checks, 'token_rows': token_rows}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'pass', 'rows': len(token_rows), 'max_prompt': max(x['prompt_tokens'] for x in token_rows),
                      'max_total': max(x['total_tokens'] for x in token_rows)}))


if __name__ == '__main__':
    main()
