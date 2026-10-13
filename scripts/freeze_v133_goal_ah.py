"""Freeze CPU-verified AH inputs and evaluation inventory, without GPU approval."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_v133_goal_ah import ROOT, OUT, Z, AG, rows, sha, stable, write_once, documents
import run_v133_goal_ae_corpus_repair as ae

def main():
    materials=rows(OUT/'judgments-reviewed.jsonl');tokens=rows(OUT/'judgment-training-inputs.jsonl')
    byid={r['judgment_id']:r for r in tokens}
    if len(materials)!=64 or set(byid)!={r['judgment_id'] for r in materials}:raise ValueError('Token material mismatch')
    for r in materials:
        t=byid[r['judgment_id']]
        if not t['within_budget'] or len(t['prompt_token_ids'])>2048 or len(t['action_token_ids'])>128:raise ValueError('Token budget exceeded')
        old=documents(r['actual_prompt']);new=documents(t['rendered_input'])
        if [(d['document_id'],d['text']) for d in old]!=[(d['document_id'],d['text']) for d in new]:raise ValueError('Actual evidence was changed during auxiliary rendering')
    original=rows(Z/'training-inputs.jsonl');cfg=json.loads((Z/'config.json').read_text(encoding='utf8'))
    original_ids={r['id'] for r in original}
    order={str(s):cfg['order'][str(s)] for s in [42,43]}
    if len(original)!=128 or any(len(o)!=128 or set(o)!=original_ids for o in order.values()):raise ValueError('Original trajectory order mismatch')
    original_tokens=sum(len(a['action_token_ids']) for r in original for a in r['actions'])
    if original_tokens!=6479:raise ValueError('Original action token count changed')
    evals=ae.frozen_cases();corpora=ae.original_paths()
    oldsplit={r['case_id']:r['split'] for r in ae.p.records(ae.p.CASES)}
    old=evals.pop('old_regression');oldcorpus=corpora.pop('old_regression')
    for key,split in [('old_main','new_eval'),('old_regression','goal_G_regression')]:
        evals[key]=[r for r in old if oldsplit[r['case_id']]==split];corpora[key]=oldcorpus
    confirm=rows(AG/'confirmation-cases-snapshot.jsonl')
    evals['ag_regression']=[dict(case_id=r.get('id',r.get('case_id')),question=r['question'],answer=r['answer'],support_titles=r.get('support_titles',[])) for r in confirm]
    corpora['ag_regression']=OUT/'ag-regression-corpus-snapshot.json'
    expected={'goal_z_dev':64,'old_main':16,'old_regression':16,'goal_y_regression':48,'aa_regression':64,'ag_regression':64}
    if {k:len(v) for k,v in evals.items()}!=expected:raise ValueError('Evaluation inventory differs')
    evaluation=[]
    for name,cases in evals.items():
        target=OUT/f'evaluation-cases-{name}.jsonl'
        if target.exists():
            if rows(target)!=cases:raise ValueError('Previously written CPU evaluation inventory changed')
        else:write_once(target,cases,jsonl=True)
        evaluation.append(dict(split=name,count=len(cases),cases=str((OUT/f'evaluation-cases-{name}.jsonl').relative_to(ROOT)),
            corpus=str(corpora[name].relative_to(ROOT)),corpus_sha256=sha(corpora[name])))
    for cases in evals.values():
        if original_ids&{r['case_id'] for r in cases}:raise ValueError('Evaluation ID in training')
        questions={ae.q.normalize_answer(r['question']) for r in cases}
        if questions&{ae.q.normalize_answer(r['question']) for r in original}:raise ValueError('Evaluation question in training')
    binding={str(s):[dict(update=2*(i+1),judgment_id=r['judgment_id']) for i,r in enumerate(sorted(materials,key=lambda r:stable(str(s)+':'+r['judgment_id'])))] for s in [42,43]}
    paths=[Z/'training-inputs.jsonl',OUT/'judgments-reviewed.jsonl',OUT/'judgment-training-inputs.jsonl',
           OUT/'material-review-main.jsonl',OUT/'model-identity.json',ROOT/'scripts/run_v133_goal_q_search.py',
           ROOT/'src/mini_agent_rl/model/local.py',ROOT/'src/mini_agent_rl/tools/search.py',
           ROOT/'scripts/run_v133_goal_ah_entity_relation.py',ae.q.CFG,
           ROOT/'runs/v133-goal-aa-audit-confirmation/product-decision.json']
    paths.extend(OUT/f'evaluation-cases-{n}.jsonl' for n in evals)
    freeze=dict(goal='AH',source_adapter='/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final',
        base='/root/autodl-tmp/qwen_08b',adapter_sha256='d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52',
        base_sha256='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
        arms=['A','B'],seeds=[42,43],updates_per_run=128,total_updates=512,order=order,judgment_binding=binding,
        original_trajectories=128,original_supervised_tokens=original_tokens,judgments=64,
        judgment_supervised_tokens=sum(len(r['action_token_ids']) for r in tokens),
        optimization=dict(learning_rate=5e-6,betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clip=1),
        loss='A=Lt; B=Lt+0.25Lj on even updates. Lt=sum action NLL / all action tokens in original trajectory; Lj=judgment NLL / judgment target tokens.',
        mask='All input/history/tool/padding masked; action JSON and im_end only supervised. No evaluation retention loss.',
        product=json.loads(ae.q.CFG.read_text(encoding='utf8'))['product'],runtime_judgment=False,
        evaluation=evaluation,total_real_trajectories=1360,fixed_context_calls=80,
        conditional_confirmation_trajectories=192,confirmation_route='B only if both seeds qualify and beat A',
        gpu_time_limit=None,stage_timeout=None,automatic_retry=False,gpu_authorized=False,
        files={str(p.relative_to(ROOT)):sha(p) for p in paths})
    write_once(OUT/'training-freeze.json',freeze)
    write_once(OUT/'cpu-freeze-check.json',dict(passed=True,actual_visible_evidence_preserved=True,
        training_eval_id_question_overlap=0,max_input_tokens=max(t['input_tokens'] for t in tokens),
        max_target_tokens=max(t['target_tokens'] for t in tokens),
        train_support_page_overlap_with_evaluation='not yet audited beyond IDs/questions; disclose before approval',
        gpu_authorized=False))
    print(json.dumps({k:freeze[k] for k in ['original_trajectories','original_supervised_tokens','judgments','judgment_supervised_tokens','total_real_trajectories']}))

if __name__=='__main__':main()
