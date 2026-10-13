"""Record frozen recovery shortage without reopening material collection."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/v133-goal-as-frozen-recovery-inventory-v1"
SOURCE=ROOT/"runs/v133-goal-ai-evidence-feasibility/closure-v3/gate-decision-v4.json"

def main():
    value=json.loads(SOURCE.read_text())
    if value["passed"]:raise RuntimeError("existing recovery gate passed: AS training must be prepared")
    ar=json.loads((ROOT/"runs/v133-goal-ar-constrained-evidence-v1/final-status.json").read_text())
    if ar["research_parent"]!="AM42A":raise RuntimeError("research parent changed")
    OUT.mkdir(exist_ok=False)
    status=dict(goal="AS",status="completed_skipped_existing_material_shortage",skipped=True,
        counts=value["counts"],checks=value["checks"],source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        new_generations=0,new_searches=0,updates=0,new_data=0,research_parent="AM42A",decision="continue_AT_no_data_expansion")
    (OUT/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
    (OUT/"REPORT.md").write_text("# Goal AS：既有恢复材料资格\n\n按冻结记录检查后跳过训练。预算内恢复1/24，门槛8/24；完整恢复对10，门槛16；关系方向/范围错配3例，门槛4。真实失败前缀10和实体错配5例通过，但不能补足其他缺口。\n\n不扩池、不补采样、不再审材料、不将回答正例计作完整恢复对。生成、搜索、更新均0，研究父版本仍AM42-A。继续AT有界动作可达性检查。原AI结论与文件全部保留。\n",encoding="utf-8")
    print(json.dumps(status))
if __name__=="__main__":main()
