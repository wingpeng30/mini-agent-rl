"""Main-agent finite dispute decisions; source reviews are preserved unchanged."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
DECISIONS={
('42-A','musique:2hop__698577_851686'):('unsupported_by_visible_evidence','Mr. Lawrence is an unrelated writer, not a supported work of Sanders. This establishes missing authorship, not a source contradiction proving that no such titled work could exist.'),
('42-A','musique:2hop__442516_595885'):('unsupported_by_visible_evidence','Vietnam belongs to another river. The named-after river/country chain for Mara Region is absent; do not classify absent target-chain evidence as factual contradiction.'),
('42-A','musique:2hop__67057_52871'):('unsupported_by_visible_evidence','1943 is Japanese withdrawal from Guadalcanal, not the requested US postwar departure; the requested departure relation is missing.'),
('42-A','musique:2hop__655505_110949'):('unsupported_by_visible_evidence','1991 is a song year; performer birth date is absent. Lack of the birthday relation is unsupported evidence, not independently demonstrated factual falsity.'),
('42-A','musique:2hop__78756_198548'):('unsupported_by_visible_evidence','Warburton voices a park ranger, not the target actor-spouse chain. Requested spouse relation is missing.'),
('42-A','5ab7eb625542993667794049'):('unsupported_by_visible_evidence','Only the facility is identified as a prison; the department is an agency. Preserve the inherited unsupported shared-type judgment under the same rubric.'),
('42-A','g010'):('unsupported_by_visible_evidence','Actress belongs to Edie. Chuck promoter/manager text does not establish shared actress occupation, and absence of an additional occupation is insufficient to prove it false.'),
('43-A','musique:2hop__698577_851686'):('unsupported_by_visible_evidence','The military-history work is attributed to Timothy Williams, not Sanders. Requested Sanders-authorship edge absent; no source disproves joint authorship, so unsupported rather than factual contradiction.'),
('43-A','musique:2hop__67057_52871'):('unsupported_by_visible_evidence','The date belongs to the Japanese Guadalcanal evacuation, not the required US postwar departure.'),
('43-A','2wiki:22a62e140bde11eba7f7acde48001122'):('unsupported_by_visible_evidence','1396 belongs to father John I; mother Martha’s death date is not given. This new answer is unsupported, not source-disproved.'),
('43-A','musique:2hop__593688_292889'):('success','Apply the already frozen baseline interpretation: club’s home country, not national-team representation. Pa Gai plays ASC HLM; its home stadium is explicitly Dakar, Senegal.'),
}
def main():
    actual={}
    for label in ['42-A','43-A']:
        for p in OUT.glob(f'actual-inputs-*-{label}.jsonl'):
            for r in map(json.loads,p.read_text(encoding='utf8').splitlines()):actual[(label,r['case_id'])]=r
    result=[]
    for (label,cid),(category,reason) in DECISIONS.items():
        r=actual[(label,cid)];result.append(dict(model=label,case_id=cid,category=category,reason=reason,
            actual_final_prompt_sha256=r['calls'][-1]['prompt_sha256'],answer=r['answer'],
            evidence_complete=category=='success' if cid=='musique:2hop__593688_292889' else None,
            reviewer='main agent',scope='One finite source-only dispute pass; no gold edits or new experiments.'))
    path=OUT/'candidate-main-adjudication-v1.jsonl';data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result).encode('utf8')
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    print('Main adjudications',len(result))
if __name__=='__main__':main()
