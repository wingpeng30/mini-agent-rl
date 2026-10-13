"""Six fixed-weight local inference checks; never trains or alters old experiments."""
import asyncio
import json
from pathlib import Path
from uuid import uuid4
from mini_agent_rl.web.server import DemoService, ROOT, RunRequest, read
from mini_agent_rl.retrieval.local import config

CASES = ("musique:2hop__796221_18803", "2wiki:f299dd600bdd11eba7f7acde48001122", "musique:2hop__96194_78303")


async def main():
    folder = Path(config()["run_dir"]) / ("agent-smoke-" + uuid4().hex)
    folder.mkdir(parents=True)
    service = DemoService(read(ROOT / "configs/demo-web.json"), runtime_dir=folder)
    service.retrieval_engine().search("warmup", "dense-v1")
    service.load_sync("goal-z")
    service.model_id, service.state = "goal-z", "ready"
    questions = {r["case_id"]: r["question"] for r in
        (json.loads(line) for line in (ROOT / "runs/v133-goal-bd-final-regression-v1/evaluation-cases.jsonl").read_text(encoding="utf-8").splitlines())}
    results = []
    try:
        for case_id in CASES:
            for mode in ("bm25-v2", "hybrid-rrf-v1"):
                request = RunRequest(question=questions[case_id], model_id="goal-z", retriever=mode)
                job_id = uuid4().hex
                service.jobs[job_id] = dict(id=job_id, state="queued", events=[])
                await service.run(job_id, request)
                job = service.jobs[job_id]
                if job["state"] != "complete":
                    raise RuntimeError(job)
                result = dict(case_id=case_id, **job["result"])
                results.append(result)
                print(json.dumps(dict(case_id=case_id, retriever=mode, answer=result["answer"],
                    reason=result["reason"], searches=result["searches"], elapsed_seconds=result["elapsed_seconds"])), flush=True)
        summary = dict(kind="six fixed-weight local Agent smoke checks; no new semantic scoring", runs=results,
            complete=len(results) == 6, output_dir=str(folder))
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(complete=True, summary=str(folder / "summary.json"))), flush=True)
    finally:
        service.release_model()


if __name__ == "__main__":
    asyncio.run(main())
