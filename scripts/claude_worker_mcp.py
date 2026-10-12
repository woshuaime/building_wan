#!/usr/bin/env python3
"""Minimal stdio MCP server that delegates approved execution tasks to Claude.

The server has no third-party runtime dependency.  It calls Anthropic's Messages
API directly and exposes a small, project-scoped tool set to the worker model.
Secrets are read from the process environment and are never written to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("CLAUDE_WORKER_ROOT", os.getcwd())).resolve()
DEFAULT_MODEL = os.environ.get("CLAUDE_WORKER_MODEL", "claude-sonnet-5")
API_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
MAX_ROUNDS = 20
MAX_OUTPUT = 30_000
READ_ONLY_TOP_LEVEL = {"data", "models", "checkpoints", "cache", "outputs"}
DENIED_EXECUTABLES = {
    "rm", "rmdir", "dd", "mkfs", "shutdown", "reboot", "poweroff",
    "git-reset", "git-clean", "git-push", "scancel", "sbatch",
}


def _json_line(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not _inside(resolved, ROOT):
        raise ValueError(f"Path escapes project root: {path_value}")
    return resolved


def _allowed_roots(values: list[str] | None) -> list[Path]:
    if not values:
        return [ROOT]
    roots = [_resolve(value) for value in values]
    return roots


def _assert_allowed(path: Path, allowed: list[Path]) -> None:
    if not any(path == root or _inside(path, root) for root in allowed):
        raise ValueError(f"Path is outside the approved scope: {path.relative_to(ROOT)}")


def _assert_writable(path: Path, allowed: list[Path]) -> None:
    _assert_allowed(path, allowed)
    relative = path.relative_to(ROOT)
    if relative.parts and relative.parts[0] in READ_ONLY_TOP_LEVEL:
        raise ValueError(f"Writes are blocked in protected directory: {relative.parts[0]}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool_definitions(allow_commands: bool) -> list[dict[str, Any]]:
    tools = [
        {
            "name": "list_files",
            "description": "List files under an approved project directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 0, "maximum": 5},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_file",
            "description": "Read a UTF-8 project file and return its SHA-256 for safe writes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_text",
            "description": "Search UTF-8 project files for a literal text string.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string", "minLength": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["path", "query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_file",
            "description": "Create or replace one UTF-8 file. Existing files require the SHA-256 returned by read_file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    ]
    if allow_commands:
        tools.append(
            {
                "name": "run_command",
                "description": "Run one non-interactive command without a shell in an approved project directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 900},
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


def _execute_worker_tool(
    name: str,
    args: dict[str, Any],
    allowed: list[Path],
    allow_commands: bool,
    allow_training: bool,
) -> str:
    if name == "list_files":
        base = _resolve(args["path"])
        _assert_allowed(base, allowed)
        depth = int(args.get("max_depth", 2))
        if not base.exists():
            return "ERROR: path does not exist"
        results: list[str] = []
        for path in sorted(base.rglob("*")):
            try:
                rel_depth = len(path.relative_to(base).parts)
            except ValueError:
                continue
            if rel_depth <= depth:
                suffix = "/" if path.is_dir() else ""
                results.append(str(path.relative_to(ROOT)) + suffix)
            if len(results) >= 500:
                results.append("... truncated ...")
                break
        return "\n".join(results)

    if name == "read_file":
        path = _resolve(args["path"])
        _assert_allowed(path, allowed)
        data = path.read_bytes()
        if len(data) > 2_000_000:
            return "ERROR: file exceeds 2 MB read limit"
        text = data.decode("utf-8")
        lines = text.splitlines(keepends=True)
        start = max(1, int(args.get("start_line", 1)))
        end = min(len(lines), int(args.get("end_line", len(lines))))
        body = "".join(lines[start - 1 : end])
        return f"sha256={_sha256(data)} lines={len(lines)} showing={start}-{end}\n{body}"

    if name == "search_text":
        base = _resolve(args["path"])
        _assert_allowed(base, allowed)
        query = args["query"]
        limit = int(args.get("max_results", 50))
        paths = [base] if base.is_file() else base.rglob("*")
        matches: list[str] = []
        for path in paths:
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if query in line:
                        matches.append(f"{path.relative_to(ROOT)}:{number}:{line[:500]}")
                        if len(matches) >= limit:
                            return "\n".join(matches)
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(matches) if matches else "No matches"

    if name == "write_file":
        path = _resolve(args["path"])
        _assert_writable(path, allowed)
        expected = args.get("expected_sha256")
        if path.exists():
            current = path.read_bytes()
            if not expected:
                return "ERROR: expected_sha256 is required when replacing an existing file"
            if _sha256(current) != expected:
                return "ERROR: file changed since it was read; read it again before writing"
        elif expected:
            return "ERROR: expected_sha256 was supplied but the file does not exist"
        data = args["content"].encode("utf-8")
        if len(data) > 2_000_000:
            return "ERROR: write exceeds 2 MB limit"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".claude-worker-tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return f"WROTE {path.relative_to(ROOT)} bytes={len(data)} sha256={_sha256(data)}"

    if name == "run_command":
        if not allow_commands:
            return "ERROR: command execution was not approved for this delegation"
        argv = args["argv"]
        executable = Path(argv[0]).name.lower()
        joined = " ".join(argv).lower()
        normalized = executable.replace("_", "-")
        if normalized in DENIED_EXECUTABLES or "--force" in argv:
            return f"ERROR: destructive command is blocked: {executable}"
        if executable in {"sh", "bash"}:
            if len(argv) < 2 or argv[1].startswith("-"):
                return "ERROR: shell snippets are blocked; only an approved project script may be run"
            script = _resolve(argv[1])
            _assert_allowed(script, allowed)
            if not allow_training and any(word in joined for word in ("train", "launch", "submit", "start_")):
                return "ERROR: training-like scripts require allow_training=true"
        if not allow_training and any(word in joined for word in ("train", "sbatch", "torchrun", "deepspeed")):
            return "ERROR: training-like commands require allow_training=true"
        cwd = _resolve(args.get("cwd", "."))
        _assert_allowed(cwd, allowed)
        timeout = int(args.get("timeout_seconds", 120))
        child_env = os.environ.copy()
        child_env.pop("ANTHROPIC_API_KEY", None)
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=timeout,
                shell=False,
            )
            output = f"exit_code={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            return output[:MAX_OUTPUT]
        except subprocess.TimeoutExpired as exc:
            return f"ERROR: command timed out after {timeout}s\n{str(exc)[:2000]}"
        except OSError as exc:
            return f"ERROR: could not start command: {exc}"

    return f"ERROR: unknown tool {name}"


def _call_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the Codex/SSH environment")
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "user-agent": "codex-claude-worker/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"Claude API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Claude API connection failed: {exc.reason}") from exc


def _delegate(args: dict[str, Any]) -> dict[str, Any]:
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task is required")
    allowed = _allowed_roots(args.get("allowed_paths"))
    allow_commands = bool(args.get("allow_commands", False))
    allow_training = bool(args.get("allow_training", False))
    if allow_training and not allow_commands:
        raise ValueError("allow_training=true requires allow_commands=true")

    if bool(args.get("dry_run", False)):
        return {
            "status": "ready" if os.environ.get("ANTHROPIC_API_KEY", "").strip() else "api_key_missing",
            "project_root": str(ROOT),
            "model": str(args.get("model") or DEFAULT_MODEL),
            "allowed_paths": [str(path.relative_to(ROOT)) or "." for path in allowed],
            "allow_commands": allow_commands,
            "allow_training": allow_training,
            "api_called": False,
            "note": "Dry-run only; no Claude API request or project mutation occurred.",
        }

    system = f"""You are Claude Execution Worker inside a GPT Codex-led workflow.
Project root: {ROOT}

Implement only the concrete task approved by the user and delegated by GPT. GPT remains the planner and final reviewer.
Rules:
- Stay inside the approved paths and task scope. Inspect before editing.
- Never reveal, print, store, or request API keys or other secrets.
- Treat data/, models/, checkpoints/, cache/, and outputs/ as read-only through file tools.
- Never delete datasets/checkpoints, rewrite Git history, force-push, or make unrelated refactors.
- Do not claim success without evidence. Distinguish static checks from GPU/runtime validation.
- Do not start training unless the delegation explicitly permits it. Prefer a short smoke test before long work.
- At the end, report files changed, commands run, validation results, failures, and work left for GPT review.
"""
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    tools = _tool_definitions(allow_commands)
    transcript: list[dict[str, Any]] = []
    model = str(args.get("model") or DEFAULT_MODEL)

    for round_number in range(1, MAX_ROUNDS + 1):
        response = _call_anthropic(
            {
                "model": model,
                "max_tokens": 8192,
                "system": system,
                "messages": messages,
                "tools": tools,
            }
        )
        content = response.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_results: list[dict[str, Any]] = []
        text_blocks: list[str] = []
        for block in content:
            if block.get("type") == "text":
                text_blocks.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                started = time.monotonic()
                try:
                    result = _execute_worker_tool(tool_name, tool_input, allowed, allow_commands, allow_training)
                    is_error = result.startswith("ERROR:")
                except Exception as exc:  # Return bounded errors to the model, not secrets or tracebacks.
                    result = f"ERROR: {type(exc).__name__}: {exc}"
                    is_error = True
                transcript.append(
                    {
                        "round": round_number,
                        "tool": tool_name,
                        "input": tool_input,
                        "result": result[:4000],
                        "is_error": is_error,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("id"),
                        "content": result,
                        "is_error": is_error,
                    }
                )
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue
        return {
            "status": "completed",
            "model": model,
            "rounds": round_number,
            "summary": "\n".join(text_blocks).strip(),
            "tool_activity": transcript,
            "review_required": True,
        }
    return {
        "status": "stopped",
        "reason": f"maximum tool rounds reached ({MAX_ROUNDS})",
        "tool_activity": transcript,
        "review_required": True,
    }


MCP_TOOL = {
    "name": "claude_execute",
    "description": (
        "Delegate an already approved implementation task to Claude, then return all activity for GPT review. "
        "Do not call for planning, analysis, review, or before the user explicitly says to execute."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Concrete approved objective, scope, constraints, and success criteria."},
            "allowed_paths": {"type": "array", "items": {"type": "string"}, "description": "Project-relative paths Claude may access."},
            "allow_commands": {"type": "boolean", "default": False},
            "allow_training": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "default": False},
            "model": {"type": "string", "description": "Optional Claude model override."},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def _handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        requested = request.get("params", {}).get("protocolVersion", "2025-06-18")
        result = {
            "protocolVersion": requested,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "claude-worker", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": [MCP_TOOL]}
    elif method == "tools/call":
        params = request.get("params", {})
        if params.get("name") != MCP_TOOL["name"]:
            raise ValueError(f"Unknown MCP tool: {params.get('name')}")
        outcome = _delegate(params.get("arguments", {}))
        result = {"content": [{"type": "text", "text": json.dumps(outcome, ensure_ascii=False, indent=2)}]}
    elif method == "ping":
        result = {}
    elif method and method.startswith("notifications/"):
        return None
    else:
        raise ValueError(f"Unsupported MCP method: {method}")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> int:
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            response = _handle(request)
            if response is not None and "id" in request:
                _json_line(response)
        except Exception as exc:
            request_id = None
            try:
                request_id = request.get("id")  # type: ignore[union-attr]
            except Exception:
                pass
            _json_line(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
