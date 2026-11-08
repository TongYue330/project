import json
import sqlite3
import tempfile
import unittest
import time
import threading
from unittest.mock import patch
from pathlib import Path

from app.agent_harness import Harness, ScriptedModel, calculate_statistics, query_history


class AgentHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db = root / "app.db"
        conn = sqlite3.connect(self.db)
        conn.executescript("""
            CREATE TABLE question (id INTEGER PRIMARY KEY, input_type TEXT, content TEXT, image_path TEXT, created_at TEXT);
            CREATE TABLE answer (id INTEGER PRIMARY KEY, question_id INTEGER, content TEXT, steps_json TEXT, confidence REAL, is_low_confidence INTEGER, created_at TEXT);
            INSERT INTO question VALUES (1, 'text', 'SQLite 数据库如何保存记录', NULL, '2026-11-01 10:00:00');
            INSERT INTO answer VALUES (1, 1, '使用 SQLite 保存', '[]', 0.90, 0, '2026-11-01 10:00:01');
            INSERT INTO question VALUES (2, 'text', '请解释快速排序复杂度', NULL, '2026-11-02 10:00:00');
            INSERT INTO answer VALUES (2, 2, '平均复杂度为 O(n log n)', '[]', 0.40, 1, '2026-11-02 10:00:01');
        """)
        conn.commit(); conn.close()
        self.log = root / "runs.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def harness(self, model, confirm=None, **kwargs):
        return Harness(db_path=self.db, run_log=self.log, **kwargs).run("统计历史答疑记录", model, confirm=confirm)

    def test_normal_complete(self):
        model = ScriptedModel([
            {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
            {"type": "final", "status": "completed", "answer": "ok", "statistics": {"total": 2}},
        ])
        result = self.harness(model)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["statistics"]["total"], 2)

    def test_parameter_error_is_returned_and_bounded(self):
        model = ScriptedModel([
            {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 99}},
            {"type": "final", "status": "failed", "answer": "invalid", "statistics": {}},
        ])
        result = self.harness(model)
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(result["retries"], 1)

    def test_tool_exception(self):
        model = ScriptedModel([
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": [{"bad": 1}]}},
        ])
        result = self.harness(model, max_retries=0)
        self.assertEqual(result["status"], "failed")
        self.assertIn("retry limit", result["reason"])

    def test_step_limit(self):
        model = ScriptedModel([{"type": "tool_call", "tool": "query_history", "arguments": {"limit": 1}}] * 5)
        result = self.harness(model, max_steps=2, max_retries=0)
        self.assertEqual(result["status"], "step_limit")

    def test_confirmation_agree_and_reject(self):
        common = [
            {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
            {"type": "tool_call", "tool": "export_statistics", "arguments": {"statistics": "$last_statistics"}},
            {"type": "final", "status": "completed", "answer": "ok", "statistics": {"total": 2}},
        ]
        operations = []
        def agree(operation):
            operations.append(operation)
            return True
        agreed = self.harness(ScriptedModel(common), confirm=agree)
        self.assertEqual(agreed["status"], "completed")
        self.assertTrue(any(c["tool"] == "export_statistics" for c in agreed["tool_calls"]))
        self.assertEqual(operations[0]["path"], f"logs/agent_exports/{agreed['task_id']}.json")
        rejected = self.harness(ScriptedModel(common), confirm=lambda operation: False, max_retries=0)
        self.assertEqual(rejected["status"], "confirmation_denied")
        self.assertFalse(any(c["tool"] == "export_statistics" for c in rejected["tool_calls"]))

    def test_schema_directly_rejects_out_of_range_limit(self):
        with self.assertRaises(Exception):
            query_history({"limit": 0}, self.db)

    def test_model_timeout(self):
        result = self.harness(ScriptedModel([{}], delay=0.05), model_timeout=0.005)
        self.assertEqual(result["status"], "timeout")

    def test_fake_statistics_are_rejected(self):
        model = ScriptedModel([
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": [{"id": 999, "question": "fake"}]}},
        ])
        result = self.harness(model, max_retries=0)
        self.assertEqual(result["status"], "failed")
        self.assertIn("latest query", result["reason"])

    def test_unknown_tool(self):
        result = self.harness(ScriptedModel([{"type": "tool_call", "tool": "run_shell", "arguments": {}}]), max_retries=0)
        self.assertEqual(result["status"], "failed")
        self.assertIn("allowlist", result["reason"])

    def test_correct_tool_arguments_directly(self):
        result = query_history({"limit": 20}, self.db, self.db)
        stats = calculate_statistics({"records": result["records"]})
        self.assertEqual((stats["total"], stats["low_confidence"], stats["average_confidence"]), (2, 1, 0.65))

    def test_tool_exception_is_fed_back(self):
        from app.agent_harness import TOOL_REGISTRY, ToolExecutionError
        model = ScriptedModel([{"type":"tool_call", "tool":"query_history", "arguments":{}}])
        with patch.dict(TOOL_REGISTRY["query_history"], {"handler": lambda *a: (_ for _ in ()).throw(ToolExecutionError("injected database failure"))}):
            result = self.harness(model, max_retries=0)
        self.assertIn("injected database failure", result["reason"])

    def test_failed_request_and_error_are_in_next_context(self):
        class InspectingModel:
            def __init__(self):
                self.messages = None
            def propose(self, messages):
                if self.messages is None:
                    self.messages = []
                    return {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 99}}
                self.messages = list(messages)
                return {"type": "final", "status": "failed", "answer": "bad limit", "statistics": {}}
        model = InspectingModel()
        self.harness(model)
        self.assertEqual([m["role"] for m in model.messages], ["user", "assistant", "tool"])
        self.assertIn('"limit": 99', model.messages[1]["content"])
        self.assertIn("limit must be an integer", model.messages[2]["content"])

    def test_final_cannot_claim_completion_without_tools(self):
        result = self.harness(ScriptedModel([{"type":"final", "status":"completed", "answer":"done", "statistics":{"total":2}}]))
        self.assertEqual(result["status"], "failed")
        self.assertIn("not executed", result["reason"])

    def test_final_statistics_must_match_tool_result(self):
        result = self.harness(ScriptedModel([
            {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
            {"type": "final", "status": "completed", "statistics": {"total": 999}},
        ]))
        self.assertEqual(result["status"], "failed")
        self.assertIn("do not match", result["reason"])

    def test_date_and_database_scope_are_checked(self):
        from app.agent_harness import ToolValidationError
        with self.assertRaises(ToolValidationError):
            query_history({"since": "invalid"}, self.db, self.db)
        with self.assertRaises(ToolValidationError):
            query_history({}, self.db.parent / "other.db", self.db)

    def test_confirmation_timeout_does_not_execute_export(self):
        from app.agent_harness import TOOL_REGISTRY
        sequence = [
            {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
            {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
            {"type": "tool_call", "tool": "export_statistics", "arguments": {"statistics": "$last_statistics"}},
        ]
        def slow_confirmation(operation):
            time.sleep(0.1)
            return True
        with patch.dict(TOOL_REGISTRY["export_statistics"], {"handler": lambda *a: self.fail("export executed after timeout")}):
            result = self.harness(ScriptedModel(sequence), confirm=slow_confirmation, tool_timeout=0.04)
        self.assertEqual(result["status"], "timeout")
        self.assertNotIn("export_statistics", [call["tool"] for call in result["tool_calls"]])

    def test_cancel_stops_the_run(self):
        harness = Harness(db_path=self.db, run_log=self.log)
        model = ScriptedModel([{"type":"tool_call", "tool":"query_history", "arguments":{}}], delay=0.2)
        timer = threading.Timer(0.02, harness.cancel)
        timer.start()
        try:
            result = harness.run("统计历史答疑记录", model)
        finally:
            timer.join()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(harness.current_state["status"], "cancelled")

    def test_log_omits_raw_question_and_answer(self):
        model = ScriptedModel([
            {"type":"tool_call", "tool":"query_history", "arguments":{"limit":20}},
            {"type":"tool_call", "tool":"calculate_statistics", "arguments":{"records":"$last_query.records"}},
            {"type":"final", "status":"completed", "answer":"ok", "statistics":{"total":2}},
        ])
        self.harness(model)
        log = self.log.read_text(encoding="utf-8")
        self.assertNotIn("快速排序复杂度", log)
        self.assertNotIn("使用 SQLite 保存", log)
        self.assertIn('"task_id"', log)


if __name__ == "__main__":
    unittest.main()

