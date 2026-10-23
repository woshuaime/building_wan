# Claude Worker Setup and Use

> Current project state and operational rules are in [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md), [AGENTS.md](../AGENTS.md), and [SESSION_HANDOFF.md](../SESSION_HANDOFF.md). Migration-era Git/Python statements below are historical. This maintenance session did not invoke Claude or test its API connectivity.

## What is already configured

Codex loads the project-level `claude_worker` MCP server from `.codex/config.toml`. The current local configuration starts it with `python3` resolved through PATH; it is separate from the scj training interpreter. Its only user-facing tool is `claude_execute`.

The worker uses the existing SSH proxy environment for outbound HTTPS. A6000 compute and Claude API access are separate: the GPU host does not need an independent public-network route as long as the SSH reverse proxy remains available.

## One-time API-key setup

When the Claude API key is ready, open the SSH terminal in this workspace and run:

```bash
bash scripts/configure_claude_key.sh
```

The script reads the key without echoing it, stores it at `~/.config/claude-worker/env` with mode `600`, and adds an early source line to `~/.bashrc`. The key is outside the project and outside Git. Reconnect the Codex SSH workspace once afterward so the app process inherits the variable.

To choose a different currently available Claude model, add an environment variable named `CLAUDE_WORKER_MODEL`. The scaffold currently defaults to `claude-sonnet-5`.

## Routing behavior

- “分析、规划、检查、审查、评估、比较、先不要执行”：GPT handles the request itself and does not call Claude.
- “可以，执行 / 开始做 / 按刚才方案修改”：GPT prepares a bounded handoff, calls `claude_execute`, then independently reviews the returned changes and evidence.
- Training remains blocked unless the current user request explicitly approves training and GPT sets both command and training permission for that single delegation.

## Limitations

- Natural-language routing is enforced by project instructions and the tool description, not by a hidden deterministic keyword router. GPT remains responsible for interpreting intent safely.
- The worker is intentionally small. It provides bounded file inspection/editing and non-interactive command execution, not a second persistent chat UI.
- The first real API request and the final model name cannot be validated before a working API key is available.
- Git is available; use the actual status and diff for review. API connectivity/model availability require a separately authorized live test.
