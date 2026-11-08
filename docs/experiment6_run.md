# Experiment 6 Agent and Harness

The implementation is in `app/agent_harness.py`. It uses a JSON protocol and
never evaluates model generated Python or shell commands.

## Run

```powershell
$env:DASHSCOPE_API_KEY = '<your-key>'
venv\Scripts\python.exe scripts/run_agent_demo.py --model qwen
venv\Scripts\python.exe scripts/run_agent_demo.py --model scripted
venv\Scripts\python.exe scripts/run_agent_demo.py --model scripted --export
venv\Scripts\python.exe -m unittest discover -s tests -v
```

The Qwen adapter reads `DASHSCOPE_API_KEY` from the environment. The key is
never written to source code or logs. The deterministic `ScriptedModel` is used
for repeatable local tests. Use `--export` to see the file name and impact and
answer `yes` before the JSON export is executed. A denied confirmation ends
the run without writing the file.
The confirmation preview includes the exact output path and replacement impact.
The current test suite contains 18 automated cases.

## Protocol

Tool request:

```json
{"type":"tool_call","tool":"query_history","arguments":{"limit":20}}
```

Final response:

```json
{"type":"final","status":"completed","answer":"...","statistics":{"total":7}}
```

Registered tools are `query_history`, `calculate_statistics`, and
`export_statistics`. The first two are read-only. Query opens the database in
SQLite read-only mode. Export is restricted to `logs/agent_exports`, requires a
confirmation callback, and uses an atomic file replacement. Raw question and
answer text stays inside the harness: Qwen receives only a summary of tool
results and references records through `$last_query.records`.

## Runtime limits

The default limits are five model steps, two correction retries, 20 seconds per
model call, five seconds per tool call, and 30 seconds for the complete run.
Every run receives a task identifier and writes redacted JSONL events to
`logs/agent_runs.jsonl`.
Failed tool requests and their error results are kept in the next model context
so the model can correct parameters within the configured retry limit.

Call `Harness.cancel()` to request termination. `Harness.current_state` exposes
the running task ID, step count, and final status. A timeout or cancellation
stops local waiting; it cannot guarantee that an already sent remote request
has stopped on the server.

