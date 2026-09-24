"""Token-level local preflight for the immutable v114 SFT exports."""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoProcessor
from train_v115 import rows,sha,token_preflight,token_signatures

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-path',required=True);p.add_argument('--train-data',required=True);p.add_argument('--internal-data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();train_path,internal_path=Path(a.train_data),Path(a.internal_data)
    train,internal=rows(train_path),rows(internal_path)
    processor=AutoProcessor.from_pretrained(a.model_path,local_files_only=True)
    report={'version':'v115-token-audit-v1','model_path':a.model_path,'processor_name_or_path':getattr(processor,'name_or_path',None),
        'train_sha256':sha(train_path),'internal_sha256':sha(internal_path),
        'train_preflight':token_preflight(processor,train),'internal_preflight':token_preflight(processor,internal),
        'train_signatures':token_signatures(processor,train),'internal_signatures':token_signatures(processor,internal),
        'internal_used_for_training':False}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'train':report['train_preflight'],'internal':report['internal_preflight'],'output':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
