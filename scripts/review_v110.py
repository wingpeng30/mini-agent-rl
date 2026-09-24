"""Blinded evidence review of every changed training/development answer pair."""
import argparse,json
from pathlib import Path
from run_v110 import API,ProtocolFailure,read_jsonl,save,verify,digest
from mini_agent_rl.hybrid_v110 import SQLiteAPILedger
from mini_agent_rl.training.thinking_sft import _env

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);p.add_argument('--env-file',required=True);p.add_argument('--ledger',default='runs/v110-api.sqlite');p.add_argument('--config',default='configs/v110-hybrid.json');p.add_argument('--cap',type=float,default=60);args=p.parse_args()
    inputs={r['task_id']:r for r in read_jsonl(args.inputs)};results=read_jsonl(args.results);cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));ledger=SQLiteAPILedger(args.ledger,total_cap=cfg['total_budget_cny']);api=API(cfg,ledger,_env(Path(args.env_file))['DEEPSEEK_API_KEY'],'review',args.cap)
    fingerprint=digest({'inputs':inputs,'results':results,'config':cfg,'review_code':Path(__file__).read_text(encoding='utf-8')})
    manifest=Path(args.output+'.manifest.json')
    if Path(args.output).exists() and not manifest.exists():raise RuntimeError('unmanifested_review')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint']!=fingerprint:raise RuntimeError('changed_review_inputs')
    save(manifest,{'fingerprint':fingerprint})
    rows=read_jsonl(args.output) if Path(args.output).exists() else [];done={r['key'] for r in rows}
    if len(done)!=len(rows):raise RuntimeError('duplicate_review_rows')
    expected={arm:{} for arm in ('B','C')}
    try:
        for result in results:
            row=inputs[result['task_id']]
            for arm in ('B','C'):
                answer=result.get(arm,row['original_answer'])
                if answer==row['original_answer']:continue
                docs=row['e0'] if arm=='B' else result['final_evidence'];key=digest([row['task_id'],row['question'],row['original_answer'],answer,docs])
                expected[arm][row['task_id']]=key
                if key in done:continue
                try:
                    audit=verify(api,row,docs,answer,blind=True)
                    item={'key':key,'task_id':row['task_id'],'answer':answer,'original':row['original_answer'],'arm':arm,'audit':audit,'status':'ok'}
                except ProtocolFailure as exc:item={'key':key,'task_id':row['task_id'],'arm':arm,'status':'unresolved','error':str(exc)}
                rows.append(item);done.add(key)
                with Path(args.output).open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                print(f"reviewed={len(rows)} cost={ledger.summary()['total_charged']:.4f}",flush=True)
        by={r['key']:r for r in rows};arms={}
        for arm,keys in expected.items():
            values=[by.get(key,{}) for key in keys.values()]
            arms[arm]={'expected_pairs':len(values),'reviewed_pairs':sum(bool(r) for r in values),
                'unresolved':sum(r.get('status')!='ok' for r in values),
                'supported_to_not_supported':[r.get('task_id') for r in values if r.get('status')=='ok' and r['audit']['assessments']['original']['status']=='supported' and r['audit']['assessments']['selected']['status']!='supported'],
                'selected_evidence_insufficient':[r.get('task_id') for r in values if r.get('status')=='ok' and r['audit']['assessments']['selected']['status']=='insufficient']}
        report={'pairs':len(rows),'arms':arms,'expected_pair_keys':expected,'fingerprint':fingerprint,'unresolved':sum(r['status']!='ok' for r in rows),'auxiliary_semantic_regressions':sum(r['status']=='ok' and r['audit']['assessments']['original']['status']=='supported' and r['audit']['assessments']['selected']['status']!='supported' for r in rows),'api_upper_cny':ledger.summary()['total_charged'],'same_model_auxiliary_only':True}
        save(args.report,report);print(json.dumps(report,indent=2))
    finally:api.client.close();ledger.close()
