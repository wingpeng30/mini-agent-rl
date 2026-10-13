"""CPU tokenizer citation positions. Read reviewed materials array from stdin."""
import json
import sys

def main():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen_08b',local_files_only=True)
    for row in json.load(sys.stdin):
        prompt=row['actual_prompt'];encoding=t(prompt,add_special_tokens=False,return_offsets_mapping=True)
        if encoding['input_ids']!=row['actual_input_token_ids']:raise ValueError('Source token roundtrip changed')
        refs={d['reference']:d for d in row['visible_documents']};citations=[]
        for quote in row['review']['source_quotes']:
            doc=refs[quote['reference']];escaped=json.dumps(quote['quote'],ensure_ascii=False)[1:-1]
            a,b=doc['input_char_span'];start=prompt.find(escaped,a,b)
            if start<0:raise ValueError('Citation absent from actual JSON input')
            end=start+len(escaped)
            covered=[i for i,(a,b) in enumerate(encoding['offset_mapping']) if a<end and b>start]
            citations.append(dict(reference=quote['reference'],document_id=doc['document_id'],quote=quote['quote'],
                input_char_span=[start,end],input_token_span=[min(covered),max(covered)+1],call_index=row['call_index']))
        print(json.dumps(dict(judgment_id=row['judgment_id'],case_id=row['case_id'],citations=citations),ensure_ascii=False))

if __name__=='__main__':main()
