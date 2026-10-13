"""Render Goal AG's saved training metrics without loading any model."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "runs" / "v133-goal-ag-failure-repair"


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    checks = {}
    for seed in (42, 43):
        for arm in ("A", "B"):
            label = f"seed{seed}-{arm}"
            folder = OUT / label
            rows = [json.loads(x) for x in (folder / "updates.jsonl").read_text().splitlines() if x]
            summary = json.loads((folder / "summary.json").read_text())
            assert len(rows) == 128 and [r["step"] for r in rows] == list(range(1, 129)), label
            assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient"]) for r in rows), label
            assert summary["base_unchanged"] and summary["changed_lora_tensors"] > 0, label
            assert summary["reload_max_abs"] == 0 and summary["updates"] == 128, label
            steps = [r["step"] for r in rows]
            losses = [r["loss"] for r in rows]
            smooth = [sum(losses[max(0, i-7):i+1]) / len(losses[max(0, i-7):i+1]) for i in range(128)]
            axes[0].plot(steps, smooth, label=label)
            axes[1].plot(steps, [r["gradient"] for r in rows], label=label, alpha=.75)
            checks[label] = {
                "updates": len(rows), "finite_metrics": True, "base_unchanged": True,
                "changed_lora_tensors": summary["changed_lora_tensors"],
                "reload_max_abs": summary["reload_max_abs"],
                "repair_update_count": summary["repair_update_count"],
                "original_supervision_tokens": sum(r["original_tokens"] for r in rows),
                "repair_supervision_tokens": sum(r.get("repair_tokens", 0) for r in rows),
                "elapsed_seconds": summary["elapsed_seconds"],
            }
    axes[0].set_ylabel("Loss (8-update rolling mean)")
    axes[1].set_ylabel("Gradient norm before clipping")
    axes[1].set_xlabel("Update")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(ncol=4)
    fig.suptitle("Goal AG: matched continuation and failure-repair SFT")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"training-curves.{suffix}", dpi=160)
    plt.close(fig)
    (OUT / "training-verification.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
