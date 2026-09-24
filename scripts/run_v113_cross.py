"""Replay frozen transition prompts only; never execute agent tools or load labels.

Run this script on the verified inference host. Inputs and manifest are prepared
locally, then their SHA256 values must be passed explicitly at invocation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

ADAPTER_HASHES = {
    'baseline': '3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768',
    'candidate': '31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tokens(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError('invalid_prompt_token_ids')
    return value


def token_hash(value):
    """Canonical JSON token hash used by the frozen P1 input manifest."""
    return digest(tokens(value))


def canonical_messages(messages):
    if not messages or any(m.get('role') not in ('system', 'user', 'assistant', 'tool')
                           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('invalid_messages')
    return [{'role': m['role'], 'content': m['content']} for m in messages]


def canonical_input_id(messages, prompt_token_ids):
    canonical_messages(messages)
    return token_hash(prompt_token_ids)


def verify_replay(row, actual_ids):
    expected = tokens(row['prompt_token_ids'])
    if tokens(actual_ids) != expected:
        raise ValueError('prompt_token_replay_mismatch:' + row['input_id'])
    if row['prompt_token_sha256'] != token_hash(expected):
        raise ValueError('prompt_token_hash_mismatch')


def deduplicate_inputs(rows):
    result = {}
    for original in rows:
        row = dict(original)
        row['prompt_token_ids'] = tokens(row['prompt_token_ids'])
        identity = canonical_input_id(row['messages'], row['prompt_token_ids'])
        if row.get('input_hash', row.get('input_id', identity)) != identity:
            raise ValueError('noncanonical_input_id')
        row['input_id'] = identity
        row['source_outputs'] = dict(row.get('observed', row.get('source_outputs', {})))
        row['used_document_ids'] = visible_document_ids(row['messages'])
        row.setdefault('prompt_token_sha256', token_hash(row['prompt_token_ids']))
        verify_replay(row, row['prompt_token_ids'])
        if type(row['step']) is not int or row['step'] < 0:
            raise ValueError('invalid_step')
        if identity in result:
            prior = result[identity]
            for arm, response in row.get('source_outputs', {}).items():
                existing = prior.setdefault('source_outputs', {}).get(arm)
                if existing is not None and existing['raw_text'] != response['raw_text']:
                    raise ValueError('source_output_conflict')
                prior['source_outputs'][arm] = response
        else:
            result[identity] = row
    return list(result.values())


def visible_document_ids(messages):
    ids = set()
    for message in messages:
        if message['role'] == 'tool':
            observation = json.loads(message['content'])
            for document in observation.get('result', []):
                if isinstance(document, dict) and isinstance(document.get('id'), str):
                    ids.add(document['id'])
    return sorted(ids)


def missing_pairs(rows, completed=()):
    done = {(r['input_id'], r['arm']) for r in completed}
    return [(row, arm) for row in rows for arm in ADAPTER_HASHES
            if (row['input_id'], arm) not in done and arm not in row.get('source_outputs', {})]


def output_record(row, arm, response, index, model_hash, origin):
    verify_replay(row, response['prompt_token_ids'])
    calls = response.get('tool_calls', [])
    parsed = None
    if len(calls) == 1 and calls[0]['name'] == 'search':
        query = calls[0]['arguments'].get('query')
        if isinstance(query, str) and query.strip():
            parsed = {'action': 'search', 'query': query}
    elif not calls and response.get('content', '').strip():
        parsed = {'action': 'answer', 'answer': response['content']}
    truncated = bool(response.get('usage', {}).get('output_truncated', False))
    valid = not response.get('parse_error') and parsed is not None
    hits = index.search(parsed['query'], top_k=3) if valid and parsed['action'] == 'search' else []
    used = set(row.get('used_document_ids', []))
    return {'input_id': row['input_id'], 'input_hash': row['input_id'], 'arm': arm, 'origin': origin,
            'adapter_sha256': ADAPTER_HASHES[arm], 'model_sha256': model_hash,
            'messages_sha256': digest(row['messages']), 'prompt_sha256': row['prompt_token_sha256'],
            'prompt_token_ids': row['prompt_token_ids'], 'prompt_token_sha256': row['prompt_token_sha256'],
            'seed': 42 + row['step'], 'temperature': 0,
            'action_raw': response['raw_text'], 'action_parsed': parsed,
            'valid_json_action': valid, 'output_truncated': truncated,
            'parse_error': response.get('parse_error'),
            'retrieval_results': [dict(hit, previously_used=hit['id'] in used,
                                        new_document=hit['id'] not in used) for hit in hits],
            'semantic_relevance': 'unjudged'}


def checked_path(path):
    path = Path(path).resolve()
    if any(x in str(path).lower() for x in ('independent100', 'finaltest')):
        raise ValueError('forbidden_path')
    return path


async def run(args):
    # All hash and tokenizer checks happen before constructing a GPU client.
    inputs = checked_path(args.inputs)
    manifest_path = checked_path(args.manifest)
    if file_hash(inputs) != args.inputs_sha256 or file_hash(manifest_path) != args.manifest_sha256:
        raise ValueError('local_remote_input_hash_mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['p1_inputs_sha256'] != args.inputs_sha256:
        raise ValueError('manifest_input_binding_mismatch')
    root = Path(__file__).resolve().parents[1]
    fingerprints = {}
    expected_sources = {f'reports/v112-replay2-{arm}-{kind}.json'
                        for arm in ADAPTER_HASHES for kind in ('tasks', 'states')}
    if set(manifest['source_files']) != expected_sources:
        raise ValueError('unexpected_source_reports')
    for source, expected in manifest['source_files'].items():
        source_path = root / source
        if file_hash(source_path) != expected:
            raise ValueError('source_report_hash_mismatch')
        report = json.loads(source_path.read_text(encoding='utf-8'))
        fingerprints[source] = report['fingerprint']
        del report
    reference = fingerprints['reports/v112-replay2-baseline-tasks.json']
    expected_config = {'device': 'cuda', 'dtype': 'bfloat16', 'load_in_4bit': False,
        'max_input_tokens': 2048, 'max_new_tokens': 128, 'enable_thinking': False,
        'deterministic': True, 'local_files_only': True, 'trust_remote_code': False,
        'observation_encoding': 'compact-v3'}
    if any(reference['model_config'].get(k) != v for k, v in expected_config.items()):
        raise ValueError('unexpected_frozen_model_config')
    for source, fp in fingerprints.items():
        arm = 'baseline' if '-baseline-' in source else 'candidate'
        if fp['adapter_files']['adapter_model.safetensors'] != ADAPTER_HASHES[arm]:
            raise ValueError('source_adapter_mismatch')
        for key in ('model_files', 'code', 'data', 'runner_config'):
            if fp[key] != reference[key]:
                raise ValueError('source_fingerprint_mismatch:' + key)
        config = {k: v for k, v in fp['model_config'].items() if k != 'adapter_path'}
        if config != {k: v for k, v in reference['model_config'].items() if k != 'adapter_path'}:
            raise ValueError('source_model_config_mismatch')
    manifest.update(model_files=reference['model_files'], model_config=reference['model_config'],
        adapter_config_sha256={arm: fingerprints[f'reports/v112-replay2-{arm}-tasks.json']['adapter_files']['adapter_config.json'] for arm in ADAPTER_HASHES},
        corpus_sha256=reference['data']['data/evidence-sft-v112/train-corpus.json'])
    for name, expected in reference['code'].items():
        if name.startswith('src/mini_agent_rl/') and file_hash(root / name) != expected:
            raise ValueError('runtime_code_hash_mismatch:' + name)
    rows = deduplicate_inputs([json.loads(line) for line in inputs.read_text(encoding='utf-8').splitlines()])
    if len(rows) != 170:
        raise ValueError('expected_170_unique_inputs')
    if len(missing_pairs(rows)) != 109:
        raise ValueError('expected_109_missing_arm_inputs')
    model = checked_path(args.model_path)
    adapter = checked_path(args.adapter_path)
    for name, expected in manifest['model_files'].items():
        path = (model / name).resolve()
        if not path.is_relative_to(model) or file_hash(path) != expected:
            raise ValueError('model_file_hash_mismatch:' + name)
    if file_hash(adapter / 'adapter_model.safetensors') != ADAPTER_HASHES[args.arm]:
        raise ValueError('frozen_adapter_hash_mismatch')
    if file_hash(adapter / 'adapter_config.json') != manifest['adapter_config_sha256'][args.arm]:
        raise ValueError('adapter_config_hash_mismatch')
    corpus_path = checked_path(args.corpus)
    if file_hash(corpus_path) != manifest['corpus_sha256']:
        raise ValueError('corpus_hash_mismatch')
    from transformers import AutoProcessor
    from mini_agent_rl.domain import Message
    from mini_agent_rl.model.context import budget_messages
    from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools.search import _BM25Index
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    def render(messages):
        return processor.apply_chat_template(canonical_messages(messages), tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True,
            return_tensors='pt')['input_ids'][0].tolist()
    for row in rows:
        # Frozen transitions store the pre-budget conversation. Inference
        # applies compact-v3 once before rendering; replay the exact effective
        # prompt after that same runtime transformation below.
        ms = [Message(**m) for m in row['messages']]
        budget = budget_messages(ms, 2048, lambda mm: len(render([m.model_dump() for m in mm])),
            lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
            observation_encoding='compact-v3')
        verify_replay(row, render([m.model_dump() for m in budget.messages]))
    index = _BM25Index(json.loads(corpus_path.read_text(encoding='utf-8')))
    output = checked_path(args.output)
    if output in (inputs, manifest_path, corpus_path) or output.is_relative_to(model) or output.is_relative_to(adapter):
        raise ValueError('output_overlaps_input')
    prior = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    known = {r['input_id']: r for r in rows}
    done = set()
    model_hash = digest(manifest['model_files'])
    for record in prior:
        key = record['input_id'], record['arm']
        if key in done or key[0] not in known or key[1] not in ADAPTER_HASHES:
            raise ValueError('invalid_checkpoint_key')
        if record['model_sha256'] != model_hash or record['adapter_sha256'] != ADAPTER_HASHES[key[1]]:
            raise ValueError('checkpoint_model_mismatch')
        verify_replay(known[key[0]], record['prompt_token_ids'])
        expected_row = known[key[0]]
        if (record['prompt_token_sha256'] != expected_row['prompt_token_sha256']
                or record['messages_sha256'] != digest(expected_row['messages'])
                or record['seed'] != 42 + expected_row['step'] or record['temperature'] != 0):
            raise ValueError('checkpoint_input_binding_mismatch')
        done.add(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    def append(record):
        with output.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
    client = None
    for row in rows:
        arm = args.arm
        if (row['input_id'], arm) in done:
            continue
        response = row.get('source_outputs', {}).get(arm)
        origin = 'source_reuse'
        if response is None:
            if client is None:
                running = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
                if running.stdout.strip():
                    raise ValueError('gpu_already_has_compute_processes:' + running.stdout.strip())
                if file_hash(inputs) != args.inputs_sha256 or file_hash(adapter / 'adapter_model.safetensors') != ADAPTER_HASHES[arm]:
                    raise ValueError('inputs_or_adapter_changed_before_cuda')
                cfg = dict(manifest['model_config'], model_id=str(model), adapter_path=str(adapter))
                client = TransformersModelClient(LocalModelConfig(**cfg))
            generated = await client.generate([Message(**m) for m in row['messages']], [],
                                              temperature=0, seed=42 + row['step'])
            response = generated.model_dump()
            origin = 'cross_inference'
        append(output_record(row, arm, response, index, model_hash, origin))
    return {'arm': args.arm, 'complete': True, 'inputs': len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', default='reports/v113-cross-inputs.jsonl')
    p.add_argument('--output', default='reports/v113-cross-results.jsonl')
    p.add_argument('--arm', choices=ADAPTER_HASHES, required=True)
    for name in ('manifest', 'inputs-sha256', 'manifest-sha256', 'model-path', 'adapter-path', 'corpus'):
        p.add_argument('--' + name, required=True)
    print(json.dumps(asyncio.run(run(p.parse_args()))))


if __name__ == '__main__':
    main()
