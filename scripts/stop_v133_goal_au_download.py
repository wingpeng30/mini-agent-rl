"""Preserve and stop a hung official-download process, not an experiment retry."""
import datetime,json,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v133-goal-au-frozen-reranking-v1'
def write(path,obj):
 with path.open('x',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
matched=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdecimal():continue
 try:parts=(proc/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeError):continue
 if len(parts)>=3 and 'python' in Path(parts[0]).name and parts[1]=='scripts/run_v133_goal_au.py' and parts[2]=='download':
  matched.append(dict(pid=int(proc.name),command=parts[:3]))
scene=dict(stage='download',failure_type='official_model_host_network_unreachable',
 reason='Independent remote curl: failed to connect huggingface.co:443 after5203ms, exit28. HfApi revision query remained pending with no downloaded files.',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),matched_processes=matched,
 files=[dict(name=p.name,bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()],
 generations=0,searches=0,training_updates=0,stop_whole_sequence=True,no_auto_rerun=True,
 remedy='Acquire the SAME official pinned revision on a reachable host, transfer and verify hashes before any explicit resume; no model or experiment parameter substitution.')
write(OUT/'network-diagnostic.json',scene)
if not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',scene)
for p in matched:os.kill(p['pid'],signal.SIGTERM)
manifest=json.loads((OUT/'manifest.json').read_text())
write(OUT/'final-status.json',dict(completed=False,status='infrastructure_stopped_before_model_acquisition',accepted=False,
 research_parent_adapter=manifest['parent_adapter'],research_parent_sha256=manifest['parent_sha256'],counts=dict(generation=0,search=0,update=0),AV_not_started=True))
with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:
 f.write('''# AU：官方模型获取阶段停止

CPU实现测试2项通过，原130条查询前三名与生产BM25一致，58条既有关系引用匹配原冻结语料。没有扩题、补查询或重新审核来源。

AutoDL到huggingface.co:443连接失败，独立curl检查在5203ms返回exit28。版本查询停滞，没有下载权重、没有冻结reranker身份，尚未进行GPU重排、真实搜索或训练。保存manifest、130原查询候选、58逐边依据、下载日志和停止现场；只终止匹配的下载进程。

按原规则停止全序列，AU尚未得出机制结论，AV未开始。没有因网络问题更换模型或参数、重跑采样。SSH连接正常，前面的AR/AS/AT已完成。

可行修复是通过能够访问BAAI官方HuggingFace仓库的机器获取相同revision的safetensors和tokenizer，再上传AutoDL核验哈希；计算仍全部在AutoDL。需要先完成这一基础设施修复，不能把缺少模型权重当成reranker负结果。
''')
print(json.dumps(scene),flush=True)
