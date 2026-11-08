"""Run the local deterministic Agent/Harness demonstration."""
import json
import sys
from app.agent_harness import Harness, ScriptedModel, DEFAULT_DB

sys.stdout.reconfigure(encoding="utf-8")


plans = [
    {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
    {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
    {"type": "final", "status": "completed", "answer": "已完成历史答疑统计。", "statistics": {"total": "$last_statistics.total"}},
]
result = Harness(db_path=DEFAULT_DB).run("查询历史答疑记录并生成统计摘要", ScriptedModel(plans))
print(json.dumps(result, ensure_ascii=False, indent=2))

