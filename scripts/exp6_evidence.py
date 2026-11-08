"""Generate redacted, reproducible evidence for the experiment report."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_harness import Harness, ScriptedModel, query_history, ToolValidationError, ToolExecutionError, TOOL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "exp6_case_runs.jsonl"
OUT = ROOT / "logs" / "exp6_case_summary.json"


def plans(export=False):
    steps = [
        {"type":"tool_call", "tool":"query_history", "arguments":{"limit":20}},
        {"type":"tool_call", "tool":"calculate_statistics", "arguments":{"records":"$last_query.records"}},
    ]
    if export:
        steps.append({"type":"tool_call", "tool":"export_statistics", "arguments":{"statistics":"$last_statistics"}})
    steps.append({"type":"final", "status":"completed", "answer":"统计完成", "statistics":{"total":"$last_statistics.total"}})
    return steps


def run_case(name, sequence, **options):
    confirmation = options.pop("confirmation", None)
    seen = []

    def confirm(operation):
        seen.append(operation)
        return confirmation

    result = Harness(run_log=LOG, **options).run(name, ScriptedModel(sequence), confirm=confirm if confirmation is not None else None)
    return {"case":name, "task_id":result["task_id"], "status":result["status"], "reason":result["reason"],
            "steps":result["steps"], "retries":result["retries"], "elapsed_ms":result["elapsed_ms"],
            "statistics":result["statistics"], "tool_names":[call["tool"] for call in result["tool_calls"]],
            "confirmation_requests":seen}


with patch.dict(TOOL_REGISTRY["query_history"], {"handler": lambda *args: (_ for _ in ()).throw(ToolExecutionError("injected database failure"))}):
    tool_failure = run_case("工具异常", [{"type":"tool_call","tool":"query_history","arguments":{"limit":1}}], max_retries=0)

cases = [
    run_case("正常完成", plans()),
    run_case("参数错误", [{"type":"tool_call","tool":"query_history","arguments":{"limit":99}}], max_retries=0),
    run_case("未知工具", [{"type":"tool_call","tool":"run_shell","arguments":{}}], max_retries=0),
    tool_failure,
    run_case("步数上限", plans(), max_steps=2),
    run_case("确认同意", plans(export=True), confirmation=True),
    run_case("确认拒绝", plans(export=True), confirmation=False),
]
try:
    query_history({"limit":0})
except ToolValidationError as exc:
    cases.append({"case":"工具直测错误参数", "status":"rejected", "reason":str(exc)})
good = query_history({"limit":1})
cases.append({"case":"工具直测正确参数", "status":"completed", "count":good["count"]})
OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for case in cases:
    print(case["case"], case["status"], case.get("reason", ""))
print("evidence:", OUT)
