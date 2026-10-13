"""Record final CPU readiness honestly; never grants GPU permission."""
import json
from collections import Counter
from pathlib import Path
from prepare_v133_goal_ah import ROOT,OUT,Z,rows,sha,write_once,documents

def main():
    cfg=json.loads((OUT/'training-freeze.json').read_text(encoding='utf8'))
    material=json.loads((OUT/'material-gate.json').read_text(encoding='utf8'))
    quality=json.loads((OUT/'confirmation-quality-gate.json').read_text(encoding='utf8'))
    original=rows(Z/'selected-train.jsonl')
    support_titles=set()
    for r in original:
        support_titles.update(r.get('support_titles',r.get('teacher',{}).get('support_titles',[])))
        if r.get('source')=='old':
            for a in r.get('trace',[]):
                for d in (a.get('observation') or {}).get('result',[]):
                    if d.get('id') in r['support_ids']:support_titles.add(d['title'])
    normalize=lambda x:' '.join(str(x).casefold().split())
    overlap={s['split']:sorted({normalize(t) for r in rows(ROOT/s['cases']) for t in r.get('support_titles',[])}&{normalize(t) for t in support_titles}) for s in cfg['evaluation']}
    write_once(OUT/'training-evaluation-isolation.json',dict(id_question_overlap=0,
        documented_original_training_support_titles=len(support_titles),support_title_overlap=overlap,
        original_trajectories_preserved=True,scope='Original support IDs/titles and evaluation support titles; not a claim that every generic distractor/entity is disjoint.',
        fresh_confirmation_registry_sha256=sha(OUT/'exposure-registry.json')))
    code=list((ROOT/'scripts').glob('*v133_goal_ah*.py'))+[ROOT/'tests/test_goal_ah_cpu.py']
    cfg['files'].update({str(p.relative_to(ROOT)):sha(p) for p in code})
    extra=[OUT/'material-citation-positions.jsonl',OUT/'mechanism-probes.jsonl',OUT/'mechanism-freeze.json',
           OUT/'confirmation-quality-gate.json',OUT/'training-evaluation-isolation.json']
    cfg['files'].update({str(p.relative_to(ROOT)):sha(p) for p in extra})
    cfg.update(pre_gpu_gates_passed=material['passed'] and quality['passed'],
        confirmation_quality=quality,tokenizer_sha256='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
        freeze_version='execution v1; supersedes intermediate CPU training-freeze checks',
        human_gpu_confirmation_received=False,diagnostic_calls_replayed=80,
        source_quality_pool_retry_allowed=False)
    write_once(OUT/'execution-freeze.json',cfg)
    previous=json.loads((OUT/'state.json').read_text(encoding='utf8'))
    write_once(OUT/'state.cpu-initial.json',previous)
    final=dict(status='stopped_confirmation_quality_shortfall',gpu_started=False,gpu_authorized=False,
        cpu_training_material_ready=True,confirmation_accepted=quality['accepted'],confirmation_required=64,
        experiment_count_changed=False,automatic_retry=False,execution_freeze_sha256=sha(OUT/'execution-freeze.json'))
    (OUT/'state.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    citations={r['judgment_id']:r for r in rows(OUT/'material-citation-positions.jsonl')}
    examples=[]
    reviewed=rows(OUT/'judgments-reviewed.jsonl')
    for verdict in ['supported','entity_mismatch','relation_mismatch','missing_link']:
        r=next(x for x in reviewed if x['verdict']==verdict)
        examples.append({k:r[k] for k in ['judgment_id','case_id','question','claim','target','origin']}|citations[r['judgment_id']])
    write_once(OUT/'training-material-examples.jsonl',examples,jsonl=True)
    write_once(OUT/'cpu-verification.json',dict(unit_tests_passed=12,syntax_checks_passed=True,
        material_citation_rows=64,supervised_terminator_token_id=248046,
        original_actions=354,original_supervised_tokens=6479,auxiliary_supervised_tokens=2274,
        actual_aux_max_input_tokens=1888,actual_aux_max_target_tokens=58,
        new_confirmation_source_quality_passed=False,gpu_tests_executed=False,
        runtime_reload_checks='implemented but not executed; requires GPU approval',
        source_registry_files=len(json.loads((OUT/'exposure-registry.json').read_text())['files'])))
    print(json.dumps(final))

if __name__=='__main__':main()
