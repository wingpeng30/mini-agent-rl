"""Preserve zero-GPU CPU-check failure and version only its API-field fix."""
from pathlib import Path
import json
import shutil
import run_v133_goal_an as an

def main():
    old=an.ROOT/"runs/v133-goal-an-hard-state-sft-v1"
    cfg=an.read(old/"manifest.json")
    failure=an.read(old/"experiment-failure.json")
    if failure["stage"]!="cpu-preflight" or failure["error_type"]!="AttributeError" or "thinking" not in failure["error"]:
        raise RuntimeError("unexpected failure; stop instead of amendment")
    if any(failure[k]!=0 for k in ("updates","generations","searches")) or (old/"gpu-usage.jsonl").exists():
        raise RuntimeError("revision allowed only before any GPU experiment")
    archives={"runner":old/"runner-before-cpu-fix.py","reviewer":old/"reviewer-before-cpu-fix.py",
              "tests":old/"tests-before-cpu-fix.py"}
    for name,r in cfg["files"].items():
        p=archives.get(name,Path(r["path"]))
        if an.sha(p)!=r["sha256"]:raise RuntimeError("original frozen file changed: "+name)
    an.am.approved()
    an.OUT.mkdir(exist_ok=False)
    for name in ("hard-state-admission.json","state-schedule.json","partial-probes.jsonl","complete-probes.jsonl","training-fit-probes.jsonl"):
        shutil.copyfile(old/name,an.OUT/name)
        assert an.sha(old/name)==an.sha(an.OUT/name)
        cfg["files"][name]["path"]=str(an.OUT/name)
    for name in ("runner","tests"):
        cfg["files"][name]["sha256"]=an.sha(cfg["files"][name]["path"])
    cfg["cpu_revision"]=dict(previous_run=str(old),previous_manifest_sha256=an.sha(old/"manifest.json"),
        reason="CPU check maps product thinking to LocalModelConfig.enable_thinking; experiment values unchanged",
        previous_updates=0,previous_generations=0,previous_searches=0)
    an.write(an.OUT/"manifest.json",cfg)
    an.write(an.OUT/"status.json",dict(status="cpu_preflight_pending",gpu_generations=0,successful_updates=0))
    (an.OUT/"EXECUTION_REVISION.md").write_text("# AN CPU预检修订\n\n保留v1 CPU失败现场；GPU调用、搜索及成功更新均为0。\n\n只修正thinking/enable_thinking的预检字段映射，添加对应CPU回归测试。34状态、引用、探针、顺序、参数、损失及门槛字节一致，未重新审核或替换材料。GPU仍等待新manifest确认。\n",encoding="utf-8")
    print(json.dumps(dict(revised=True,manifest_sha256=an.sha(an.OUT/"manifest.json"),gpu_calls=0)))

if __name__=="__main__":main()
