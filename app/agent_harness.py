"""A bounded single-agent runtime for course Q&A history statistics.

The module deliberately uses a small JSON protocol instead of executing model
generated Python.  A model can request only one of the registered tools and
the harness validates every argument before dispatch.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
from urllib import error as urlerror
from urllib import request as urlrequest


LOGGER = logging.getLogger("course_agent")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "app.db"
EXPORT_DIR = PROJECT_ROOT / "logs" / "agent_exports"
RUN_LOG = PROJECT_ROOT / "logs" / "agent_runs.jsonl"

MAX_RECORDS = 20
MAX_STEPS = 5
MAX_RETRIES = 2
MODEL_TIMEOUT_SECONDS = 20.0
TOOL_TIMEOUT_SECONDS = 5.0
TOTAL_TIMEOUT_SECONDS = 30.0


class AgentError(Exception):
    """An expected, user-visible agent failure."""


class ToolValidationError(AgentError):
    pass


class ToolExecutionError(AgentError):
    pass


class OperationTimeout(AgentError):
    pass


class ConfirmationDenied(AgentError):
    pass


class ConfirmationRequired(AgentError):
    def __init__(self, message: str, operation: Mapping[str, Any]):
        super().__init__(message)
        self.operation = dict(operation)


def _safe_json(value: Any, limit: int = 1200) -> Any:
    """Keep logs useful without recording credentials or unbounded text."""
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v, limit) for k, v in value.items() if str(k).lower() not in {"key", "api_key", "authorization", "token", "password", "question", "answer", "content", "records", "top_keywords"}}
    if isinstance(value, list):
        return [_safe_json(v, limit) for v in value[:MAX_RECORDS]]
    if isinstance(value, str):
        return value[:limit]
    return value


def _iso_date(value: Any, field: str) -> Optional[str]:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ToolValidationError(f"{field} must be an ISO date string")
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ToolValidationError(f"{field} must use YYYY-MM-DD") from exc


def query_history(arguments: Mapping[str, Any], db_path: Path = DEFAULT_DB, allowed_db: Optional[Path] = None) -> Dict[str, Any]:
    """Read at most 20 Q&A records from the project database."""
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("arguments must be an object")
    keyword = arguments.get("keyword", "")
    if not isinstance(keyword, str) or len(keyword) > 100:
        raise ToolValidationError("keyword must be a string of at most 100 characters")
    limit = arguments.get("limit", 10)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RECORDS:
        raise ToolValidationError("limit must be an integer from 1 to 20")
    since = _iso_date(arguments.get("since"), "since")
    until = _iso_date(arguments.get("until"), "until")
    if since and until and since > until:
        raise ToolValidationError("since must not be later than until")

    allowed_db = (allowed_db or DEFAULT_DB).resolve()
    requested_db = Path(db_path).resolve()
    if requested_db != allowed_db:
        raise ToolValidationError("database path is outside the permitted project database")
    if not requested_db.exists():
        raise ToolExecutionError("project database does not exist")

    clauses: list[str] = []
    params: list[Any] = []
    if keyword:
        clauses.append("(q.content LIKE ? OR a.content LIKE ?)")
        pattern = f"%{keyword}%"
        params.extend([pattern, pattern])
    if since:
        clauses.append("date(q.created_at) >= date(?)")
        params.append(since)
    if until:
        clauses.append("date(q.created_at) <= date(?)")
        params.append(until)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT q.id, q.content AS question, q.input_type, q.created_at,
               a.content AS answer, a.confidence, a.is_low_confidence
        FROM question q LEFT JOIN answer a ON a.question_id = q.id
        {where}
        ORDER BY q.created_at DESC, q.id DESC LIMIT ?
    """
    params.append(limit)
    try:
        conn = sqlite3.connect(requested_db.as_uri() + "?mode=ro", uri=True, timeout=TOOL_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ToolExecutionError(f"database query failed: {exc}") from exc
    return {"records": rows, "count": len(rows), "filters": {"keyword": keyword, "since": since, "until": until}}


def calculate_statistics(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Calculate deterministic, inspectable statistics for query_history output."""
    if not isinstance(arguments, Mapping) or not isinstance(arguments.get("records"), list):
        raise ToolValidationError("records must be an array returned by query_history")
    records = arguments["records"]
    if len(records) > MAX_RECORDS:
        raise ToolValidationError("records cannot contain more than 20 items")
    for item in records:
        if not isinstance(item, Mapping) or "id" not in item or "question" not in item:
            raise ToolValidationError("each record must contain id and question")
    try:
        confidences = [float(r["confidence"]) for r in records if r.get("confidence") is not None]
        if any(not 0 <= value <= 1 for value in confidences):
            raise ValueError("confidence outside 0..1")
    except (TypeError, ValueError) as exc:
        raise ToolValidationError("confidence must be a number from 0 to 1") from exc
    low = sum(1 for r in records if bool(r.get("is_low_confidence")))
    type_counts: Dict[str, int] = {}
    for record in records:
        kind = str(record.get("input_type") or "unknown")
        type_counts[kind] = type_counts.get(kind, 0) + 1
    keyword_counts: Dict[str, int] = {}
    for record in records:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]{1,}|[\u4e00-\u9fff]{2,}", str(record.get("question") or "").lower())
        for word in set(words):
            keyword_counts[word] = keyword_counts.get(word, 0) + 1
    top_keywords = sorted(keyword_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    return {
        "total": len(records),
        "low_confidence": low,
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "input_type_counts": dict(sorted(type_counts.items())),
        "top_keywords": [{"keyword": word, "count": count} for word, count in top_keywords],
        "record_ids": [int(record["id"]) for record in records],
    }


def export_statistics(arguments: Mapping[str, Any], task_id: str, project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Write a JSON export below logs/agent_exports after confirmation."""
    if not isinstance(arguments, Mapping) or not isinstance(arguments.get("statistics"), Mapping):
        raise ToolValidationError("statistics must be an object")
    output_name = arguments.get("output_name", f"{task_id}.json")
    if not isinstance(output_name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", output_name) or not output_name.endswith(".json"):
        raise ToolValidationError("output_name must be a simple .json filename")
    export_root = (Path(project_root) / "logs" / "agent_exports").resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    target = (export_root / output_name).resolve()
    if export_root not in target.parents:
        raise ToolValidationError("export path is outside the permitted export directory")
    payload = {"task_id": task_id, "statistics": _safe_json(arguments["statistics"]), "created_at": datetime.now().isoformat(timespec="seconds")}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return {"path": str(target.relative_to(Path(project_root).resolve())), "bytes": target.stat().st_size}


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "query_history": {"handler": query_history, "read_only": True, "confirmation": False},
    "calculate_statistics": {"handler": calculate_statistics, "read_only": True, "confirmation": False},
    "export_statistics": {"handler": export_statistics, "read_only": False, "confirmation": True},
}


def tool_schemas() -> Dict[str, Dict[str, Any]]:
    return {
        "query_history": {"purpose": "Read project Q&A history", "required": [], "fields": {"keyword": "string <=100", "since": "YYYY-MM-DD", "until": "YYYY-MM-DD", "limit": "integer 1..20"}},
        "calculate_statistics": {"purpose": "Calculate statistics from query_history records", "required": ["records"], "fields": {"records": "array <=20"}},
        "export_statistics": {"purpose": "Export statistics after user confirmation", "required": ["statistics"], "fields": {"statistics": "object", "output_name": "simple .json filename"}},
    }


def resolve_references(value: Any, context: Mapping[str, Any]) -> Any:
    """Resolve the small $last_tool.records convenience used by the demo model."""
    if isinstance(value, str) and value.startswith("$"):
        current: Any = context
        for part in value[1:].split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ToolValidationError(f"unknown context reference: {value}")
            current = current[part]
        return current
    if isinstance(value, Mapping):
        return {k: resolve_references(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, context) for v in value]
    return value


class ScriptedModel:
    """Deterministic model substitute used for local tests and demonstrations."""

    def __init__(self, plans: Iterable[Mapping[str, Any]], delay: float = 0.0):
        self.plans = list(plans)
        self.index = 0
        self.delay = delay

    def propose(self, messages: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        if self.index >= len(self.plans):
            return {"type": "final", "status": "completed", "answer": "No further action", "statistics": {}}
        plan = self.plans[self.index]
        self.index += 1
        return plan


class QwenAgentModel:
    """Optional Qwen adapter. The key is read only from DASHSCOPE_API_KEY."""

    def __init__(self, timeout: float = MODEL_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

    def propose(self, messages: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise AgentError("DASHSCOPE_API_KEY is not configured")
        system = {"role": "system", "content": (
            "你是单Agent，目标是查询本项目答疑历史并生成统计摘要。只输出一个合法JSON对象，不要Markdown。"
            "先调用query_history，参数keyword为可选字符串、limit为1到20的整数；再调用calculate_statistics，参数必须为"
            "{\"records\":\"$last_query.records\"}。工具结果会作为后续user消息回传。"
            "最终输出{\"type\":\"final\",\"status\":\"completed\",\"answer\":\"...\","
            "\"statistics\":{\"total\":\"$last_statistics.total\"}}。"
            "调用工具时输出{\"type\":\"tool_call\",\"tool\":\"工具名\",\"arguments\":{...}}。"
            "只允许query_history、calculate_statistics、export_statistics；导出需用户确认。"
        )}
        api_messages = [system]
        for message in messages:
            if message.get("role") == "tool":
                result = json.loads(message.get("content", "{}"))
                # Keep raw question/answer history local. The model chooses
                # references to the records held in the harness instead.
                api_messages.append({"role": "user", "content": "工具执行结果：" + json.dumps(_safe_json(result), ensure_ascii=False)})
            else:
                api_messages.append({"role": message["role"], "content": str(message.get("content", ""))})
        payload = json.dumps({"model": self.model, "messages": api_messages, "temperature": 0.1, "stream": False}).encode()
        req = urlrequest.Request(self.url, data=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (urlerror.URLError, TimeoutError) as exc:
            raise AgentError(f"model connection failed: {exc}") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AgentError("model response was not valid JSON") from exc
        if not isinstance(parsed, Mapping) or parsed.get("type") not in {"tool_call", "final"}:
            raise AgentError("model response must specify type tool_call or final")
        return parsed


class Harness:
    """Runs one bounded agent task and records an auditable JSONL event."""

    def __init__(self, *, db_path: Path = DEFAULT_DB, run_log: Path = RUN_LOG, max_steps: int = MAX_STEPS,
                 max_retries: int = MAX_RETRIES, model_timeout: float = MODEL_TIMEOUT_SECONDS,
                 tool_timeout: float = TOOL_TIMEOUT_SECONDS, total_timeout: float = TOTAL_TIMEOUT_SECONDS):
        self.db_path = Path(db_path).resolve()
        self.run_log = Path(run_log)
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.model_timeout = model_timeout
        self.tool_timeout = tool_timeout
        self.total_timeout = total_timeout
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.current_state: Dict[str, Any] = {"status": "idle", "steps": 0}

    def cancel(self) -> None:
        """Request termination of the next wait/step; running threads may remain."""
        self._cancel.set()

    def _append_log(self, event: Mapping[str, Any]) -> None:
        event = {"time": datetime.now().isoformat(timespec="milliseconds"), **event}
        self.run_log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.run_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_safe_json(event), ensure_ascii=False) + "\n")

    def _call_timeout(self, function: Callable[[], Any], timeout: float) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(function)
        try:
            deadline = time.monotonic() + timeout
            while True:
                if self._cancel.is_set():
                    future.cancel()
                    raise ConfirmationDenied("run cancelled by user; already running background work may continue")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FutureTimeout()
                try:
                    return future.result(timeout=min(remaining, 0.05))
                except FutureTimeout:
                    if time.monotonic() >= deadline:
                        raise
        except FutureTimeout as exc:
            future.cancel()
            raise OperationTimeout(f"operation timed out after {timeout:g}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _check_final(self, response: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[bool, str]:
        if response.get("status") != "completed":
            return False, "model reported failure"
        statistics = response.get("statistics")
        if not isinstance(statistics, Mapping) or "total" not in statistics:
            return False, "final response did not include checked statistics"
        if state.get("last_statistics") is None:
            return False, "statistics tool was not executed"
        reported_total = statistics.get("total")
        if isinstance(reported_total, str) and re.fullmatch(r"[0-9]+", reported_total):
            reported_total = int(reported_total)
        if isinstance(reported_total, bool) or not isinstance(reported_total, int):
            return False, "final total must be an integer"
        if reported_total != state["last_statistics"].get("total"):
            return False, "final statistics do not match tool result"
        for field, value in statistics.items():
            if field != "total" and state["last_statistics"].get(field) != value:
                return False, f"final {field} does not match tool result"
        return True, "success criteria satisfied"

    def run(self, task: str, model: Any, *, confirm: Optional[Callable[[Mapping[str, Any]], bool]] = None) -> Dict[str, Any]:
        if not isinstance(task, str) or not task.strip() or len(task) > 1000:
            raise ToolValidationError("task must contain 1..1000 characters")
        task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        started = time.perf_counter()
        messages: list[Mapping[str, Any]] = [{"role": "user", "content": task}]
        state: Dict[str, Any] = {"last_query": None, "last_statistics": None, "tool_calls": []}
        retries = 0
        final: Dict[str, Any] = {}
        status = "failed"
        reason = "agent did not finish"
        steps = 0
        self.current_state = {"task_id": task_id, "status": "running", "steps": 0}
        self._append_log({"event": "run_started", "task_id": task_id, "task_sha256": hashlib.sha256(task.encode()).hexdigest(), "started_at": datetime.now().isoformat(timespec="seconds")})
        try:
            while steps < self.max_steps:
                if self._cancel.is_set():
                    status, reason = "cancelled", "run cancelled by user"
                    break
                elapsed = time.perf_counter() - started
                if elapsed >= self.total_timeout:
                    status, reason = "timeout", f"total runtime exceeded {self.total_timeout:g}s"
                    break
                steps += 1
                self.current_state["steps"] = steps
                try:
                    proposal = self._call_timeout(lambda: model.propose(messages), min(self.model_timeout, max(0.01, self.total_timeout - elapsed)))
                    if not isinstance(proposal, Mapping):
                        raise AgentError("model response must be a JSON object")
                    original_proposal = dict(proposal)
                    proposal = resolve_references(proposal, state)
                    if proposal.get("type") == "final":
                        if time.perf_counter() - started >= self.total_timeout:
                            status, reason = "timeout", f"total runtime exceeded {self.total_timeout:g}s"
                            break
                        ok, check_reason = self._check_final(proposal, state)
                        final = dict(proposal)
                        if ok:
                            status, reason = "completed", check_reason
                            final["statistics"] = state["last_statistics"]
                        else:
                            status, reason = "failed", check_reason
                        break
                    if proposal.get("type") != "tool_call":
                        raise AgentError("model response type is invalid")
                    tool_name = proposal.get("tool")
                    if tool_name not in TOOL_REGISTRY:
                        raise ToolValidationError("tool is not in the allowlist")
                    arguments = resolve_references(proposal.get("arguments", {}), state)
                    if not isinstance(arguments, Mapping):
                        raise ToolValidationError("arguments must be an object")
                    allowed_fields = set(tool_schemas()[tool_name]["fields"])
                    if set(arguments) - allowed_fields:
                        raise ToolValidationError("unknown tool argument field")
                    metadata = TOOL_REGISTRY[tool_name]
                    if tool_name == "calculate_statistics" and (state["last_query"] is None or arguments.get("records") != state["last_query"]["records"]):
                        raise ToolValidationError("statistics records must match the latest query result")
                    if tool_name == "export_statistics" and (state["last_statistics"] is None or arguments.get("statistics") != state["last_statistics"]):
                        raise ToolValidationError("export statistics must match the latest calculation")
                    operation = {"tool": tool_name, "arguments": _safe_json(arguments), "task_id": task_id,
                                 "impact": "Write or replace a JSON file in logs/agent_exports"}
                    if metadata["confirmation"]:
                        if confirm is None or not bool(self._call_timeout(lambda: confirm(operation), min(self.tool_timeout, max(0.01, self.total_timeout - (time.perf_counter() - started))))):
                            raise ConfirmationDenied("user confirmation was denied or unavailable")
                    started_tool = time.perf_counter()
                    if tool_name == "query_history":
                        call = lambda: metadata["handler"](arguments, self.db_path, self.db_path)
                    elif tool_name == "export_statistics":
                        call = lambda: metadata["handler"](arguments, task_id, PROJECT_ROOT)
                    else:
                        call = lambda: metadata["handler"](arguments)
                    result = self._call_timeout(call, min(self.tool_timeout, max(0.01, self.total_timeout - (time.perf_counter() - started))))
                    duration_ms = round((time.perf_counter() - started_tool) * 1000)
                    state["tool_calls"].append({"tool": tool_name, "arguments": _safe_json(arguments), "result": _safe_json(result), "duration_ms": duration_ms})
                    if tool_name == "query_history":
                        state["last_query"] = result
                    if tool_name == "calculate_statistics":
                        state["last_statistics"] = result
                    messages.append({"role": "assistant", "content": json.dumps(original_proposal, ensure_ascii=False)})
                    messages.append({"role": "tool", "name": tool_name, "content": json.dumps(result, ensure_ascii=False)})
                    self._append_log({"event": "tool_succeeded", "task_id": task_id, "step": steps, "tool": tool_name, "arguments": _safe_json(arguments), "result": _safe_json(result), "duration_ms": duration_ms})
                except (OperationTimeout, ConfirmationDenied) as exc:
                    status = "timeout" if isinstance(exc, OperationTimeout) else ("cancelled" if self._cancel.is_set() else "confirmation_denied")
                    reason = str(exc)
                    self._append_log({"event": "step_error", "task_id": task_id, "step": steps, "error_type": type(exc).__name__, "error": reason, "retry": False})
                    break
                except (ToolValidationError, ToolExecutionError, ConfirmationRequired, AgentError, sqlite3.Error) as exc:
                    retries += 1
                    error_text = str(exc)
                    self._append_log({"event": "step_error", "task_id": task_id, "step": steps, "error_type": type(exc).__name__, "error": error_text, "retry": retries <= self.max_retries})
                    messages.append({"role": "tool", "name": "harness", "content": json.dumps({"error": error_text, "retry_remaining": max(0, self.max_retries - retries)}, ensure_ascii=False)})
                    if retries > self.max_retries:
                        status, reason = "failed", f"retry limit reached: {error_text}"
                        break
            else:
                status, reason = "step_limit", f"maximum steps {self.max_steps} reached"
            if steps >= self.max_steps and status == "failed" and reason == "agent did not finish":
                status, reason = "step_limit", f"maximum steps {self.max_steps} reached"
        except Exception as exc:  # keep the harness observable even for unexpected failures
            status, reason = "failed", f"unexpected harness error: {type(exc).__name__}: {exc}"
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        self.current_state = {"task_id": task_id, "status": status, "steps": steps, "elapsed_ms": elapsed_ms}
        output = {"task_id": task_id, "status": status, "reason": reason, "steps": steps, "retries": retries, "elapsed_ms": elapsed_ms,
                  "answer": final.get("answer", ""), "statistics": final.get("statistics"), "tool_calls": state["tool_calls"]}
        self._append_log({"event": "run_finished", **output})
        return output


def demo_model() -> ScriptedModel:
    return ScriptedModel([
        {"type": "tool_call", "tool": "query_history", "arguments": {"limit": 20}},
        {"type": "tool_call", "tool": "calculate_statistics", "arguments": {"records": "$last_query.records"}},
        {"type": "final", "status": "completed", "answer": "已根据历史答疑记录生成统计摘要。", "statistics": {"total": 0}},
    ])

