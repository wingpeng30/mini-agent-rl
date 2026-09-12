"""SQLite 持久化：事件只追加，保证轨迹可审计和可回放。"""
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from ..domain import Event, Rollout, RolloutGroup, Task, Transition

class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = str(path); self.conn = sqlite3.connect(self.path); self.conn.row_factory = sqlite3.Row; self.init()
    def init(self):
        self.conn.executescript("""CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL); INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version); CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS rollout_groups(id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS rollouts(id TEXT PRIMARY KEY, group_id TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS transitions(id TEXT PRIMARY KEY, rollout_id TEXT NOT NULL, step INTEGER NOT NULL, data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, rollout_id TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL);"""); self.conn.commit()
    def save_task(self, task: Task): self.conn.execute("INSERT OR REPLACE INTO tasks VALUES (?,?)", (task.id, task.model_dump_json())); self.conn.commit()
    def save_group(self, group: RolloutGroup): self.conn.execute("INSERT OR REPLACE INTO rollout_groups VALUES (?,?)", (group.id, group.model_dump_json())); self.conn.commit()
    def save_rollout(self, rollout: Rollout): self.conn.execute("INSERT OR REPLACE INTO rollouts VALUES (?,?,?,?)", (rollout.id, rollout.group_id, rollout.status.value, rollout.model_dump_json())); self.conn.commit()
    def save_transition(self, transition: Transition): self.conn.execute("INSERT OR REPLACE INTO transitions VALUES (?,?,?,?)", (transition.id, transition.rollout_id, transition.step, transition.model_dump_json())); self.conn.commit()
    def append_event(self, event: Event): self.conn.execute("INSERT INTO events VALUES (?,?,?,?)", (event.id, event.rollout_id, event.type, event.model_dump_json())); self.conn.commit()
    def get_rollout(self, rollout_id: str):
        row = self.conn.execute("SELECT data FROM rollouts WHERE id=?", (rollout_id,)).fetchone(); return Rollout.model_validate_json(row[0]) if row else None
    def list_rollouts(self, status: str | None = None):
        rows = self.conn.execute("SELECT data FROM rollouts" + (" WHERE status=?" if status else ""), ((status,) if status else ())).fetchall(); return [Rollout.model_validate_json(r[0]) for r in rows]
    def export_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for row in self.conn.execute("SELECT data FROM rollouts"):
                rollout = Rollout.model_validate_json(row[0]); transitions = [Transition.model_validate_json(x[0]).model_dump(mode="json") for x in self.conn.execute("SELECT data FROM transitions WHERE rollout_id=? ORDER BY step", (rollout.id,))]; f.write(json.dumps({"rollout": rollout.model_dump(mode="json"), "transitions": transitions}, ensure_ascii=False) + "\n")
    def close(self): self.conn.close()
