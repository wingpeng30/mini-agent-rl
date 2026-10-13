"""Post-execution CPU integrity acceptance. Never invokes a model."""
import json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v133-goal-ah-entity-relation'
def rows(p):return list(map(json.loads,p.read_text(encoding='utf8').splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cfg=json.loads((OUT/'execution-freeze.json').read_text(encoding='utf8'))
    for name,digest in cfg['files'].items():assert sha(ROOT/name)==digest,name
    approval=json.loads((OUT/'gpu-approval.json').read_text(encoding='utf8'))
    assert sha(OUT/'execution-freeze.json') in json.dumps(approval)
    remote=json.loads((OUT/'final-remote-integrity.json').read_text(encoding='utf8'));assert remote['passed']
    assert remote['source_adapter_sha256']=='d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52'
    assert remote['automatic_retries']==0 and remote['new_confirmation_outputs']==[] and not remote['experiment_failure_latch']
    stages=json.loads((OUT/'candidate-evaluation-stage.json').read_text(encoding='utf8'));assert stages['status']=='completed'
    count=0;calls=0;observations=0
    protected={r['case_id'] for r in rows(OUT/'necessary-search-protection-frozen.jsonl')};assert len(protected)==86
    for label in ['baseline','42-A','43-A','42-B','43-B']:
        reviews={r['case_id']:r for r in rows(OUT/f'final-source-review-{label}.jsonl')};assert len(reviews)==272
        actual={r['case_id']:r for f in OUT.glob(f'actual-inputs-*-{label}.jsonl') for r in rows(f)};assert len(actual)==272
        if label=='baseline':assert all(reviews[cid]['category']=='success' for cid in protected)
        for f in OUT.glob(f'eval-*-{label}.jsonl'):
            for raw in rows(f):
                count+=1;assert raw['model_calls']<=4 and raw['searches']<=3
                assert math.isfinite(raw['f1']) and math.isfinite(raw['elapsed_seconds'])
                for action in raw['trace']:
                    calls+=1;assert len(action['prompt_token_ids'])<=2048 and len(action['action_token_ids'])<=128
                a=actual[raw['case_id']];r=reviews[raw['case_id']];call=a['calls'][-1]
                assert r['actual_final_prompt_sha256']==call['prompt_sha256'] and r['answer']==raw['answer']
                assert r['entity_relation_frame'] and r['category']!='ambiguous'
                if r['category']=='success':assert r['evidence_complete'] and r['observed_relations_with_positions'] and not r['missing_relations']
                for q in r['actual_visible_source']+r['observed_relations_with_positions']:
                    s,e=q['actual_input_char_span'];escaped=json.dumps(q['quote'],ensure_ascii=False)[1:-1]
                    assert call['actual_prompt'][s:e]==escaped
                    lo,hi=q['actual_input_token_span'];assert 0<=lo<hi<=len(call['prompt_token_ids']);observations+=1
    assert count==1360 and len(rows(OUT/'paired-product-ledger.jsonl'))==1088
    probe_calls=0
    for label in ['baseline','42-A','43-A','42-B','43-B']:probe_calls+=len(rows(OUT/f'mechanism-{label}.jsonl'))
    assert probe_calls==80
    for label in ['42-A','43-A','42-B','43-B']:
        summary=json.loads((OUT/f'seed{label}/summary.json').read_text(encoding='utf8'));updates=rows(OUT/f'seed{label}/updates.jsonl')
        assert len(updates)==128 and summary['base_unchanged'] and summary['changed_lora_tensors']==192
        assert summary['reload_sentinels']==3 and summary['reload_max_abs']==0
        assert remote['candidates'][label]['adapter_sha256']==summary['adapter_sha256']
        assert json.loads((OUT/f'evaluation-exit-{label}.json').read_text(encoding='utf8'))['returncode']==0
        judgments=[u for u in updates if u['judgment_id'] is not None]
        assert len(judgments)==(64 if label.endswith('B') else 0)
    results=json.loads((OUT/'gpu-results.json').read_text(encoding='utf8'))
    assert all(not r['passed'] for r in results['development_gates'].values())
    assert all(not r['passed'] for r in results['mechanism_gates'].values())
    assert results['confirmation_calls']==0 and not results['publication_allowed']
    result=dict(passed=True,cpu_only=True,frozen_files=55,product_trajectories=1360,actual_model_calls=calls,mechanism_calls=80,
        checked_quote_positions=observations,training_updates=512,protected_cases=86,automatic_retries=0,new_confirmation_calls=0,
        source_ledger_cases=1360,paired_cases=1088,product_model_changed=False)
    (OUT/'final-cpu-integrity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
