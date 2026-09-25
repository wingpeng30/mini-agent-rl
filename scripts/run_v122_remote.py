"""Run the one preregistered v122 preflight, smoke, and formal SFT sequence."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(root: Path, model: Path, source: Path, reference: Path, output: Path) -> None:
    train_dir = root / "data/evidence-sft-v122-train-candidates/frozen"
    internal_dir = root / "data/evidence-sft-v122-internal-candidates/frozen"
    train = train_dir / "train.jsonl"
    internal = internal_dir / "internal.jsonl"
    train_gate = root / "reports/v122-train-preflight.json"
    internal_gate = root / "reports/v122-internal-preflight.json"
    config = root / "configs/v122-targeted-sft.json"
    local_tokens = root / "reports/v122-local-token-audit.json"
    output.mkdir(parents=True, exist_ok=False)
    preflight = output / "remote-preflight.json"
    subprocess.run([sys.executable, "-m", "scripts.preflight_v122",
        "--model-path", str(model), "--source-adapter", str(source),
        "--reference-adapter", str(reference), "--train", str(train),
        "--internal", str(internal), "--corpus", str(internal_dir / "internal-corpus.json"),
        "--train-gate", str(train_gate), "--internal-gate", str(internal_gate),
        "--train-manifest", str(train_dir / "manifest.json"),
        "--internal-manifest", str(internal_dir / "manifest.json"),
        "--local-token-audit", str(local_tokens), "--config", str(config),
        "--workspace", str(output.parent), "--output", str(preflight)],
        cwd=root, check=True)
    shared = ["--source-adapter", str(source), "--model-path", str(model),
        "--train", str(train), "--internal", str(internal),
        "--train-gate", str(train_gate), "--internal-gate", str(internal_gate),
        "--remote-preflight", str(preflight),
        "--token-audit", str(preflight.with_suffix(".tokens.json")),
        "--config", str(config)]
    smoke = output / "smoke"
    subprocess.run([sys.executable, "-m", "scripts.train_v122", "--phase", "smoke",
                    *shared, "--output-dir", str(smoke)], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "scripts.train_v122", "--phase", "formal",
                    *shared, "--smoke-report", str(smoke / "smoke-report.json"),
                    "--output-dir", str(output / "formal")], cwd=root, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "model", "source", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.model, args.source, args.reference, args.output)
