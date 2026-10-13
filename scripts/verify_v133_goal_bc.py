"""Final CPU archive/count verification; no inference or retrieval."""
import json
import run_v133_goal_bc as bc

def main():
    bc.frozen();out=bc.OUT
    status=bc.read(out/'final-status.json');gate=bc.read(out/'probe-gate.json')
    inputs={r['probe_id']:r for r in bc.rows(out/'probe-inputs.jsonl')}
    prior={r['probe_id']:r for r in bc.rows(bc.AW/'probe-A.jsonl')}
    actual=bc.rows(out/'engineer.jsonl')+bc.rows(out/'probe-A.jsonl')+bc.rows(out/'probe-B.jsonl')
    counts=status['counts']
    assert len(actual)==counts['primary']==74
    assert sum(r['usage'].get('context_aware_decoding',{}).get('reference_sequences',0) for r in actual)==counts['reference']==36
    assert counts['search']==0 and not list(out.glob('eval-*.jsonl'))
    assert not status['accepted'] and not gate['passed'] and status['completed']
    assert status['training_updates']==status['new_confirmation']==0
    a=bc.rows(out/'probe-A.jsonl');b=bc.rows(out/'probe-B.jsonl')
    assert len(a)==len(b)==32
    for r in actual:
        assert r['prompt_token_ids']==inputs[r['probe_id']]['prompt_token_ids']
        assert len(r['prompt_token_ids'])<=bc.cfg('A')['max_input_tokens']
        assert len(r['action_token_ids'])<=bc.cfg('A')['max_new_tokens']
        assert not bc.clipping(r['usage'])
    exact=sum(r['action_token_ids']==prior[r['probe_id']]['action_token_ids'] for r in a)
    identities={}
    for arm in ('A','B','alpha0'):
        eng=bc.read(out/f'engineer-{arm}-identity.json')
        records=[r for r in actual if r['arm']==arm]
        identities[arm]=dict(load_seconds=eng['load_seconds'],peak_memory_gb=eng['memory']['peak_allocated_gb'])
        if arm!='alpha0':
            probe=bc.read(out/f'probe-{arm}-complete.json')
            identities[arm]['load_seconds']+=probe['load_seconds']
            identities[arm]['peak_memory_gb']=max(identities[arm]['peak_memory_gb'],probe['memory']['peak_allocated_gb'])
        identities[arm]['generation_seconds_excluding_model_load']=sum(r['usage']['elapsed_generation_seconds'] for r in records)-identities[arm]['load_seconds']
    result=dict(passed=True,A_exact_archived_generation_tokens=exact,A_actual_inputs=32,B_actual_inputs=32,
      counts=counts,source_hashes_passed=True,citations_and_cache_lengths_verified_by_finalizer=True,
      no_real_search_after_gate_failure=True,costs=identities,
      report_sha256=bc.sha(out/'REPORT.md'),adjudication_freeze_sha256=bc.sha(out/'adjudication-freeze.json'))
    bc.write(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
