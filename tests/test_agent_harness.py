import json
import sqlite3
import tempfile
import unittest
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
        agreed = self.harness(ScriptedModel(common), confirm=lambda operation: True)
        self.assertEqual(agreed["status"], "completed")
        self.assertTrue(any(c["tool"] == "export_statistics" for c in agreed["tool_calls"]))
        rejected = self.harness(ScriptedModel(common), confirm=lambda operation: False, max_retries=0)
        self.assertEqual(rejected["status"], "failed")

    def test_schema_directly_rejects_out_of_range_limit(self):
        with self.assertRaises(Exception):
            query_history({"limit": 0}, self.db)


if __name__ == "__main__":
    unittest.main()

