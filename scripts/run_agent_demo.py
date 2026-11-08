"""Run the history-statistics single agent from the project root or scripts dir."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_harness import Harness, ScriptedModel, QwenAgentModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "scripted"], default="qwen")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    task = "查询最近最多20条答疑历史，统计总数、低置信度数量和平均置信度。"
    task += "统计后申请导出摘要。" if args.export else "只读查询和统计，不要导出。"
    plans = [
        {"type":"tool_call", "tool":"query_history", "arguments":{"limit":20}},
        {"type":"tool_call", "tool":"calculate_statistics", "arguments":{"records":"$last_query.records"}},
    ]
    if args.export:
        plans.append({"type":"tool_call", "tool":"export_statistics", "arguments":{"statistics":"$last_statistics"}})
    plans.append({"type":"final", "status":"completed", "answer":"已完成历史答疑统计。", "statistics":{"total":"$last_statistics.total"}})
    model = QwenAgentModel() if args.model == "qwen" else ScriptedModel(plans)
    harness = Harness()

    def confirm(operation):
        print("导出操作及影响：", json.dumps(operation, ensure_ascii=False))
        return input("确认写入以上文件？输入 yes 同意，其他输入拒绝：").strip().lower() == "yes"

    print(json.dumps(harness.run(task, model, confirm=confirm if args.export else None), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

