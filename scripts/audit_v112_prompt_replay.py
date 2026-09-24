"""Read-only audit of frozen single-batch tokens against actual runtime prompts."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoProcessor
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    p = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)
    def render(ms):
        return p.apply_chat_template([{'role':m.role,'content':m.content} for m in ms],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors='pt')['input_ids'][0].tolist()
    report = {'version':'v112','model_path':a.model_path,'splits':{},'passed':True}
    for split in ('train','validation'):
        path = root/f'data/evidence-sft-v112/{split}.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        result = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(rows),
                  'max_actual_tokens':0,'single_batch_rows':0,'failures':[]}
        for row in rows:
            ms = [Message(**m) for m in row['messages'][:-1]]
            saved = row['prompt_token_ids']
            if len(saved)==1 and isinstance(saved[0],list):
                saved=saved[0];result['single_batch_rows']+=1
            valid = bool(saved) and all(type(x) is int and x>=0 for x in saved)
            direct = render(ms)
            budget = budget_messages(ms,2048,lambda m:len(render(m)),
                lambda s:len(p.tokenizer.encode(s,add_special_tokens=False)),observation_encoding='compact-v3')
            runtime = render(budget.messages)
            result['max_actual_tokens']=max(result['max_actual_tokens'],len(direct))
            if not valid or direct!=saved or runtime!=saved or len(direct)>2048:
                result['failures'].append({'state_hash':row['state_hash'],'valid_token_shape':valid,
                    'direct_equal':direct==saved,'runtime_equal':runtime==saved,'length':len(direct)})
        report['splits'][split]=result
        report['passed'] &= not result['failures']
    out=Path(a.output)
    if out.exists():raise ValueError('output_already_exists')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
