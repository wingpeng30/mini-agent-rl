"""One-shot, train-only GPU signal probe. No parameter updates."""
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, set_seed
from evaluate_v133_goal_c_sft import normalize_answer, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v133-goal-m-readiness'
TRAIN = ROOT / 'runs/v133-goal-j-transfer/train-frozen.jsonl'
ADAPTER = ROOT / 'runs/v133-goal-j-transfer/adapter-final'
MODEL = Path('/root/autodl-tmp/qwen_08b')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(raw):
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return 'invalid', None
    if isinstance(value, dict) and set(value) == {'action', 'answer'} and value['action'] == 'answer' and isinstance(value['answer'], str) and value['answer'].strip():
        return 'answer', value['answer'].strip()
    if isinstance(value, dict) and set(value) == {'action', 'query'} and value['action'] == 'search' and isinstance(value['query'], str) and value['query'].strip():
        return 'search', value['query'].strip()
    return 'invalid', None


def main():
    cfg = json.loads((OUT / 'probe-config.json').read_text(encoding='utf-8'))
    assert sha(TRAIN) == cfg['train_sha256'] and sha(ADAPTER / 'adapter_model.safetensors') == cfg['adapter_sha256']
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 22 * 1024**3
    rows = [json.loads(x) for x in TRAIN.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 16
    assert not (OUT / 'probe-results.jsonl').exists(), 'probe already started; no automatic retry'
    state = {'status': 'running', 'completed': 0, 'planned': 64, 'config_sha256': sha(OUT / 'probe-config.json')}
    (OUT / 'probe-state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    set_seed(cfg['seed'])
    started = time.monotonic()
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    base = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda', local_files_only=True)
    model = PeftModel.from_pretrained(base, ADAPTER, local_files_only=True)
    model.eval()
    active = model.active_adapters() if callable(getattr(model, 'active_adapters', None)) else [model.active_adapter]
    assert 'default' in active
    state['active_adapters'] = active
    state['lora_modules'] = sum(1 for _, m in model.named_modules() if hasattr(m, 'lora_A') and 'default' in m.lora_A)
    assert state['lora_modules'] == 96
    times = []
    try:
        with (OUT / 'probe-results.jsonl').open('w', encoding='utf-8') as sink:
            for row in rows:
                ids = proc.apply_chat_template(row['messages'][:row['target_assistant_index']], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
                ids = list(ids[0] if ids and isinstance(ids[0], list) else ids)
                assert len(ids) <= cfg['max_prompt_tokens']
                for candidate in range(cfg['candidates_per_question']):
                    if time.monotonic() - started >= cfg['hard_cap_seconds']:
                        raise TimeoutError('probe hard cap')
                    begin = time.monotonic()
                    input_ids = torch.tensor([ids], device=model.device)
                    with torch.inference_mode():
                        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                                   do_sample=True, temperature=cfg['temperature'], top_p=cfg['top_p'],
                                                   top_k=cfg['top_k'], max_new_tokens=cfg['max_new_tokens'],
                                                   pad_token_id=proc.tokenizer.eos_token_id)
                    action_ids = generated[0, len(ids):].tolist()
                    raw = proc.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
                    kind, answer = classify(raw)
                    em = int(normalize_answer(answer) == normalize_answer(row['official_answer'])) if kind == 'answer' else 0
                    f1 = f1_score(answer, row['official_answer']) if kind == 'answer' else 0.0
                    reward = cfg['reward']['valid_answer_em'] * em + cfg['reward']['valid_answer_f1'] * f1 if kind == 'answer' else 0.0
                    result = {'case_id': row['case_id'], 'candidate': candidate, 'raw': raw, 'kind': kind,
                              'em': em, 'f1': f1, 'reward': reward, 'action_tokens': len(action_ids),
                              'prompt_tokens': len(ids), 'generation_seconds': round(time.monotonic()-begin, 3)}
                    sink.write(json.dumps(result, ensure_ascii=False) + '\n'); sink.flush()
                    times.append(result['generation_seconds']); state['completed'] += 1
                    if state['completed'] == 2:
                        projection = time.monotonic()-started + sum(times)/2 * 62 + 120
                        state['first_two_projection_seconds'] = round(projection, 3)
                        if projection > cfg['hard_cap_seconds']:
                            raise TimeoutError('first two speed projection exceeds cap')
                    (OUT / 'probe-state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        results = [json.loads(x) for x in (OUT / 'probe-results.jsonl').read_text(encoding='utf-8').splitlines()]
        groups = [[x for x in results if x['case_id'] == row['case_id']] for row in rows]
        variable = sum(len({x['reward'] for x in group}) > 1 for group in groups)
        answer_variable = sum({x['em'] for x in group if x['kind'] == 'answer'} == {0, 1} for group in groups)
        state.update({'status': 'complete', 'elapsed_seconds': round(time.monotonic()-started, 3),
                      'content_variable_groups': variable, 'answer_correctness_variable_groups': answer_variable,
                      'zero_variance_groups': 16-variable, 'kinds': {k: sum(x['kind']==k for x in results) for k in ('answer','search','invalid')},
                      'action_tokens_max': max(x['action_tokens'] for x in results),
                      'signal_pass': variable >= cfg['min_content_variable_groups'] and answer_variable >= cfg['min_answer_correctness_variable_groups']})
    except Exception as exc:
        state.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': round(time.monotonic()-started, 3)})
        raise
    finally:
        (OUT / 'probe-state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(state))


if __name__ == '__main__':
    main()
