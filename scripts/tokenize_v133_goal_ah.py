"""CPU tokenizer only; JSON array on stdin, token records on stdout."""
import json
import re
import sys

SYSTEM = ('AH_EVIDENCE_JUDGMENT. Use only the supplied tool evidence. Return JSON with '
          'verdict, facts and missing. Verdict: supported, entity_mismatch, relation_mismatch, '
          'missing_link. Facts are [entity,relation,object,Eref]; missing are '
          '[entity,needed_relation,"?"]. E1,E2,... index documents in tool-result order, '
          'including repeated documents. Insufficient evidence does not mean factually false.')

def encode(tokenizer, row):
    blocks = re.findall(r'<tool_response>.*?</tool_response>', row['actual_prompt'], re.S)
    user = ('Question: '+row['question']+'\nCandidate relations: '+
            json.dumps(row['claim'], ensure_ascii=False, separators=(',', ':'))+
            '\nActual visible tool evidence:\n'+'\n'.join(blocks))
    encoded = tokenizer.apply_chat_template(
        [dict(role='system', content=SYSTEM), dict(role='user', content=user)],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1: raise ValueError('Unexpected tokenizer batch')
        ids = ids[0]
    target = json.dumps(row['target'], ensure_ascii=False, separators=(',', ':'))
    action = tokenizer.encode(target+'<|im_end|>', add_special_tokens=False)
    if not action or action[-1] != tokenizer.convert_tokens_to_ids('<|im_end|>'):
        raise ValueError('Missing supervised terminator')
    return dict(judgment_id=row['judgment_id'], prompt_token_ids=ids,
                prompt_attention_mask=[1]*len(ids), action_token_ids=action,
                input_tokens=len(ids), target_tokens=len(action),
                within_budget=len(ids)<=2048 and len(action)<=128,
                rendered_input=tokenizer.decode(ids, skip_special_tokens=False))

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen_08b', local_files_only=True)
    for row in json.load(sys.stdin):
        print(json.dumps(encode(tokenizer,row), ensure_ascii=False))

if __name__=='__main__': main()
