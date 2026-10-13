"""Read-only report of Goal AG model paths, configs, and adapter tensor dtypes."""
import json
from collections import Counter
from pathlib import Path

import run_v133_goal_af_retrieval_validation as af
import run_v133_goal_q_search as q
import run_v133_goal_z_multihop_sft as z
from safetensors import safe_open

adapter_file = z.SOURCE / "adapter_model.safetensors"
adapter_config = json.loads((z.SOURCE / "adapter_config.json").read_text(encoding="utf-8"))
base_config_path = q.p.BASE / "config.json"
base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
dtype_counts = Counter()
sample_names = []
with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
    for key in f.keys():
        dtype_counts[str(f.get_tensor(key).dtype)] += 1
        if len(sample_names) < 4:
            sample_names.append(key)
print(json.dumps({
    "base_path": str(q.p.BASE), "base_exists": q.p.BASE.is_dir(),
    "base_model_type": base_config.get("model_type"),
    "base_safetensors": [{"name": p.name, "sha256": q.sha(p)}
                          for p in sorted(q.p.BASE.glob("*.safetensors"))],
    "adapter_path": str(z.SOURCE), "adapter_sha256": q.sha(adapter_file),
    "adapter_base_metadata": adapter_config.get("base_model_name_or_path"),
    "adapter_tensor_dtypes": dict(dtype_counts), "sample_tensor_names": sample_names,
    "published_identity": af.validate_release_identity(),
}, ensure_ascii=False, indent=2))
