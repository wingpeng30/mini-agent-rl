from transformers import AutoTokenizer
from pathlib import Path
t=AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1]/'runs/v133-goal-au-frozen-reranking-v2/reranker'),local_files_only=True)
print('class',type(t),'mro',type(t).__mro__)
print('public',[(k,hasattr(t,k)) for k in ['backend_tokenizer','_tokenizer','build_inputs_with_special_tokens','create_token_type_ids_from_sequences','pad','encode','__call__']])
print('special',t.cls_token_id,t.sep_token_id,t.model_input_names)
b=t.backend_tokenizer
print('processor',b.post_processor)
print('pair',t('Who was born here?','a body',add_special_tokens=True,truncation=False))
import tokenizers,inspect
print('encoding signature',inspect.signature(tokenizers.Encoding))
print('encoding methods',[x for x in dir(tokenizers.Encoding) if not x.startswith('_')])
