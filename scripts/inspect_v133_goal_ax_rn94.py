import run_v133_goal_ap as ap
from mini_agent_rl.model.grounded_action_audit import visible_documents
s=next(s for s in ap.rows(ap.AM/'candidate-training-states.jsonl') if s['case_id']=='musique:2hop__106336_47169')
t=ap.am.tokenizer_from_base();m=ap.restore_tool_messages(ap.parse_archived_messages(t,s['prompt_token_ids']))
for d in visible_documents(m):
 if d['document_id']==s['bridge_document_id']:print(d['text'])