"""User-authorized AU transport repair; preserve v1 and identical experiment."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import run_v133_goal_au as au
ROOT=au.ROOT;OLD=au.OUT;OUT=ROOT/'runs/v133-goal-au-frozen-reranking-v2'

class MeasuredReranker(au.FrozenReranker):
    def __init__(self,*args,**kwargs):
        start=time.monotonic();super().__init__(*args,**kwargs)
        au.ap.append(OUT/'component-load-times.jsonl',dict(component='reranker',tokenizer_only=kwargs.get('tokenizer_only',False),seconds=time.monotonic()-start))
    def rank(self,query,candidates):
        pairs,mapping,spans=self.pairs(query,candidates)
        au.ap.append(OUT/'reranker-actual-inputs.jsonl',dict(query=query,pairs=pairs,window_to_document=mapping,spans=spans,training_use_forbidden=True))
        return super().rank(query,candidates)

def measured_client_factory(original):
    def factory(*args,**kwargs):
        start=time.monotonic();client,wrapped,registry=original(*args,**kwargs)
        au.ap.append(OUT/'component-load-times.jsonl',dict(component='product_factory_and_index',seconds=time.monotonic()-start))
        start=time.monotonic();client._load()
        au.ap.append(OUT/'component-load-times.jsonl',dict(component='agent_model',seconds=time.monotonic()-start))
        return client,wrapped,registry
    return factory

def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('AU v2 already frozen')
    old=au.read(OLD/'manifest.json')
    for k,p in old['paths'].items():
        if au.sha(p)!=old['hashes'][k]:raise RuntimeError('AU unchanged source drift '+k)
    model=OUT/'reranker';info=au.read(model/'model-info.json');revision=info['sha']
    if revision!='2cfc18c9415c912f9d8155881c133215df768a70':raise RuntimeError('AU pinned official revision')
    names=['README.md','config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']
    by={r['rfilename']:r for r in info['siblings']};files={};proof=[]
    for name in names:
        p=model/name;r=by[name]
        if p.stat().st_size!=r['size']:raise RuntimeError('AU official size mismatch '+name)
        h=au.sha(p)
        if r.get('lfs'):
            if h!=r['lfs']['sha256']:raise RuntimeError('AU official LFS hash mismatch '+name)
            method='official LFS SHA256'
        else:
            data=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob!=r['blobId']:raise RuntimeError('AU official Git blob mismatch '+name)
            method='official Git object SHA1 plus recorded SHA256'
        files[name]=h;proof.append(dict(file=name,sha256=h,verified_against=method))
    for filename,key in [('offline-queries.jsonl','queries'),('query-cases.jsonl','mapping'),('approved-edges.jsonl','edges')]:
        shutil.copyfile(OLD/filename,OUT/filename);old['paths'][key]=str(OUT/filename)
    for filename in ['PLAN.md']:shutil.copyfile(OLD/filename,OUT/filename)
    au.write(OUT/'gpu-usage.jsonl',[],jsonl=True)
    au.write(OUT/'reranker-identity.json',dict(repo=au.REPO,revision=revision,files=files,official_proof=proof,transport_only_change=True))
    au.write(OUT/'NETWORK_REPAIR.md','''# AU transport-only repair

User explicitly supplied network_turbo and domestic-mirror remedy after v1 infrastructure stop. Prior failure/log/manifest retained unchanged.
Remote network_turbo+official source still SSL-timeout; network_turbo+hf-mirror.com metadata responds200. The fixed safetensors is acquired directly via this mirror; tokenizer and official verification metadata were acquired through reachable local official HuggingFace endpoint and transferred to AutoDL. The redundant slow full-model upload was cancelled and its partial archive preserved. No generated state or inference experiment was restarted.
Revision2cfc18c9415c912f9d8155881c133215df768a70. Safetensors SHA256 ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd matches official LFS; all three LFS artifacts and four Git blobs checked in remote preparation.
Only acquisition transport changed. Same parent AM,130 original queries,58 relation edges,corpus,model,windowing,scoring and original total GPU/search caps. v1 GPU/search/update0, v2 continues original allowance, not a repeated inference experiment. All experiment computation remains remote.
''')
    old.update(execution_revision='v2-transport-repair',prior_failure_preserved=True,prior_gpu_generations=0,prior_searches=0,prior_updates=0,
      user_authorization='如果无法连接huggingface，用国内镜像源，或者启用加速source /etc/network_turbo',
      same_model_and_experiment=True)
    extras=dict(resume_runner=Path(__file__),old_manifest=OLD/'manifest.json',old_failure=OLD/'experiment-failure.json',
        network_repair=OUT/'NETWORK_REPAIR.md',official_metadata=model/'model-info.json',reranker_identity=OUT/'reranker-identity.json')
    old['paths'].update({k:str(v) for k,v in extras.items()});old['hashes']={k:au.sha(v) for k,v in old['paths'].items()}
    au.write(OUT/'manifest.json',old);print(json.dumps(dict(prepared=True,revision=revision,official_files_verified=7)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','feasibility','offline','gate','evaluate']);p.add_argument('--arm',choices=['A','B']);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare()
        else:
            au.OUT=OUT;au.MODEL=OUT/'reranker'
            au.FrozenReranker=MeasuredReranker
            if a.stage=='evaluate':au.ap.q.make_client=measured_client_factory(au.ap.q.make_client)
            if a.stage=='evaluate':au.evaluate(a.arm)
            else:getattr(au,a.stage)()
    except BaseException as exc:
        if not (OUT/'experiment-failure.json').exists():au.write(OUT/'experiment-failure.json',dict(stage=a.stage,error_type=type(exc).__name__,error=str(exc),stop_whole_sequence=True,no_auto_rerun=True))
        raise
if __name__=='__main__':main()
