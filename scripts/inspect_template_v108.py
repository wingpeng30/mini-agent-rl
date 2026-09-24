"""Record whether the installed processor has a thinking-default mismatch."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]
explicit = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
default = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
report = {"version": "v108", "explicit_false_equals_default": explicit == default,
          "explicit_false_has_think": "<think>" in explicit,
          "default_has_think": "<think>" in default,
          "explicit_sha256": hashlib.sha256(explicit.encode()).hexdigest(),
          "default_sha256": hashlib.sha256(default.encode()).hexdigest()}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
