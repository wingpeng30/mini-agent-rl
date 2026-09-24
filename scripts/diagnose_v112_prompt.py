"""CPU-only comparison of frozen SFT prompt tokens and runtime re-budgeting."""
import argparse,json
from pathlib import Path
from transformers import AutoProcessor
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages

def main(model_path,data):
    row=json.loads(Path(data).read_text(encoding='utf-8').splitlines()[0])
    p=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    messages=[Message(**m) for m in row['messages'][:-1]]
    def render(ms):
        return p.apply_chat_template([{'role':m.role,'content':m.content} for m in ms],tokenize=True,add_generation_prompt=True,
                                     enable_thinking=False,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
    budget=budget_messages(messages,2048,lambda ms:len(render(ms)),lambda s:len(p.tokenizer.encode(s,add_special_tokens=False)),observation_encoding='compact-v3')
    direct=render(messages);runtime=render(budget.messages);saved=row['prompt_token_ids']
    i=next((i for i,(a,b) in enumerate(zip(saved,runtime)) if a!=b),min(len(saved),len(runtime)))
    print(json.dumps({'saved_len':len(saved),'direct_len':len(direct),'runtime_len':len(runtime),'direct_matches_saved':direct==saved,
      'runtime_matches_saved':runtime==saved,'first_difference':i,'saved_fragment':p.tokenizer.decode(saved[max(0,i-15):i+45]),
      'runtime_fragment':p.tokenizer.decode(runtime[max(0,i-15):i+45]),
      'changed_messages':[{'before':a.content,'after':b.content} for a,b in zip(messages,budget.messages) if a.content!=b.content]},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--data',required=True);a=p.parse_args();main(a.model_path,a.data)
