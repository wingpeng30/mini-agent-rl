"""Plot the four frozen Goal Z loss traces; no model or GPU is loaded."""
from __future__ import annotations

import matplotlib.pyplot as plt

import run_v133_goal_z_multihop_sft as z


def smooth(values: list[float], width: int = 16) -> list[float]:
    return [sum(values[max(0, i - width + 1):i + 1]) /
            len(values[max(0, i - width + 1):i + 1]) for i in range(len(values))]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for ax, arm in zip(axes, ("answer", "trajectory")):
        for seed in (42, 43):
            rows = z.rows(z.OUT / f"seed{seed}-{arm}/updates.jsonl")
            if len(rows) != 128:
                raise RuntimeError("Incomplete training curve")
            ax.plot([r["step"] for r in rows], smooth([r["loss"] for r in rows]),
                    label=f"seed {seed}")
        ax.set(title=f"{arm} SFT", xlabel="Update", ylabel="Mean action-token NLL")
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(z.OUT / "training-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
