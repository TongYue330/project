# Experiment 6 Agent and Harness

The implementation is in `app/agent_harness.py`. It uses a JSON protocol and
never evaluates model generated Python or shell commands.

## Run

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/run_agent_demo.py
python -m unittest discover -s tests -v
```

The Qwen adapter reads `DASHSCOPE_API_KEY` from the environment. The key is
never written to source code or logs. The deterministic `ScriptedModel` is used
for repeatable local tests.

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
`export_statistics`. The first two are read-only. Export is restricted to
`logs/agent_exports`, requires a confirmation callback, and uses an atomic file
replacement.

## Runtime limits

The default limits are five model steps, two correction retries, 20 seconds per
model call, five seconds per tool call, and 30 seconds for the complete run.
Every run receives a task identifier and writes redacted JSONL events to
`logs/agent_runs.jsonl`.

