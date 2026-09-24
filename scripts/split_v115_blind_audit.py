"""Split blinded v115 audit packets into manageable independent-review batches."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def emit(root,prefix,rows,size):
    out=[]
    for start in range(0,len(rows),size):
        path=root/f'{prefix}-{start//size+1:02d}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows[start:start+size]),encoding='utf-8')
        out.append({'path':str(path),'cases':len(rows[start:start+size]),'sha256':sha(path)})
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--packet-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    packet=Path(a.packet_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    states=readl(packet/'states-blind.jsonl');tasks=readl(packet/'changed-tasks-blind.jsonl')
    manifest=json.loads((packet/'manifest.json').read_text(encoding='utf-8'))
    if len(states)!=manifest['state_cases'] or len(tasks)!=manifest['changed_task_cases']:raise ValueError('packet_count_mismatch')
    s=emit(out,'states',states,10);t=emit(out,'changed-tasks',tasks,2)
    result={'version':'v115-audit-batches-v1','source_manifest_sha256':sha(packet/'manifest.json'),
        'state_batches':s,'changed_task_batches':t,'state_count':len(states),'changed_task_count':len(tasks)}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'state_batches','changed_task_batches'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
