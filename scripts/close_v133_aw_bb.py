"""Mechanical closure of conditional stages; no inference, training or retrieval."""
from pathlib import Path
import run_v133_goal_ay as ay
from finalize_v133_goal_ay import citations
ap=ay.ap
def main():
    ay.aw.frozen();aw=ap.read(ay.aw.OUT/'final-status.json');result=ap.read(ay.OUT/'final-status.json');gate=ap.read(ay.OUT/'material-gate.json')
    if aw['accepted'] or result['accepted'] or gate['passed']:raise RuntimeError('Cannot close accepted chain as empty')
    root=ap.ROOT/'runs/v133-aw-bb-sequential-checks-v1';root.mkdir(exist_ok=True)
    ax=ap.ROOT/'runs/v133-goal-ax-independent-reader-v1';ax.mkdir(exist_ok=True)
    source=ap.AM/'candidate-training-states.jsonl';states=ap.rows(source)
    if len(states)!=34:raise RuntimeError('AX inventory changed')
    state=next(s for s in states if s['case_id']=='musique:2hop__106336_47169')
    cite=citations(ap.am.tokenizer_from_base(),state['prompt_token_ids'],[(state['bridge_document_id'],state['bridge_quote'],['RN-94','joint developers','Nurol and Romarm'])])
    if not cite:raise RuntimeError('AX cited body not actually visible')
    ay.write(ax/'manifest.json',dict(source_sha256=ap.sha(source),fixed_states=34,failed_state_id=state['state_id'],first_hop_status='unconfirmed',reason='Joint developers based in different countries do not establish one vehicle-origin country.',citations=cite,actual_generation=0,actual_search=0,actual_updates=0,old_AM_scores_unchanged=True))
    dependencies=[ay.aw.OUT/'final-status.json',ay.OUT/'final-status.json',ay.OUT/'material-gate.json',ax/'manifest.json',ap.ROOT/'docs/v133-aw-bb-sequential-plan.md',root/'FINAL_REPORT.md']
    ay.write(root/'manifest.json',dict(completed=True,accepted_version=None,parent='AM seed42-A',parent_sha256=ap.AM_HASH,published='Goal Z',published_sha256=ap.Z_HASH,actual_generation=aw['counts']['generation']+result['generation'],actual_search=result['search'],actual_updates=0,teacher_calls=0,new_confirmation=0,stages=dict(AW='rejected_probe',AX='closed_material',AY='closed_material',AZ='not_triggered',BA='not_triggered',BB='not_triggered'),hashes={str(p.relative_to(ap.ROOT)):ap.sha(p) for p in dependencies},prior_AY_failure_preserved=True,repair_rerun_explicitly_user_authorized=True))
    for name,reason in [('v133-goal-az-refreshed-action-policy-v1','AY reliable corrections below8'),('v133-goal-ba-sequential-reproduction-v1','No accepted new method'),('v133-goal-bb-cumulative-regression-v1','No accepted cumulative candidate')]:
        folder=ap.ROOT/'runs'/name;folder.mkdir(exist_ok=True)
        ay.write(folder/'manifest.json',dict(executed=False,reason=reason,generation=0,search=0,updates=0,source_manifest_sha256=ap.sha(root/'manifest.json')))
    print('Sequential closure saved: 269 generations /65 searches /0 updates; no accepted version.',flush=True)
if __name__=='__main__':main()
