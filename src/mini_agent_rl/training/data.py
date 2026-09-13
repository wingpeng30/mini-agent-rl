"""从已验证轨迹生成 SFT 消息数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..domain import Transition


def export_sft_messages(trajectories: Iterable[list[Transition]], output: str | Path) -> int:
    """仅导出成功轨迹中的模型动作；工具 observation 保留为 tool 消息但不是训练标签。"""

    count = 0
    with Path(output).open("w", encoding="utf-8") as handle:
        for transitions in trajectories:
            if not transitions or not transitions[-1].terminated:
                continue
            messages = [m.model_dump(mode="json") for m in transitions[-1].messages]
            response = transitions[-1].response
            messages.append({"role": "assistant", "content": json.dumps({"action": "answer", "answer": response.content}, ensure_ascii=False)})
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    return count
