"""Read-only AY review packet: actual inputs, existing sources, no model calls."""
import json
import run_v133_goal_ay as ay
from mini_agent_rl.model.grounded_action_audit import visible_documents,map_documents
ap=ay.ap
def main():
    if not ap.read(ay.OUT/'collection-complete.json')['cases']==32:raise RuntimeError('Incomplete collection')
    sources={r['id']:r for r in ap.rows(ap.Z/'selected-train.jsonl')};tok=ap.am.tokenizer_from_base()
    for number,r in enumerate(ap.rows(ay.OUT/'collected-trajectories.jsonl'),1):
        source=sources[r['case_id']];last=r['trace'][-1]
        if ap.q.score(r['answer'],source['answer'])['em']:continue
        messages=ap.restore_tool_messages(ap.parse_archived_messages(tok,last['prompt_token_ids']))
        docs=visible_documents(messages)
        packet=dict(number=number,case_id=r['case_id'],question=r['question'],action=last.get('action'),answer=r['answer'],query=last.get('query'),step=last['step'],source_answer=source['answer'],chain=source.get('chain'),source_support=source['teacher']['retrieved'],actual_documents=docs,trace_actions=[{k:x.get(k) for k in ('step','action','query','answer','repeated')} for x in r['trace']])
        ap.append(ay.OUT/'review-packets.jsonl',packet)
        print(json.dumps(packet,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
