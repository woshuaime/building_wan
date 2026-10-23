# 历史迁移进度（兼容入口）

2026-10-14：当前状态统一维护在根目录 [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)，操作规则见 [AGENTS.md](../AGENTS.md)，开发审计见 [PROJECT_AUDIT_20261014.md](PROJECT_AUDIT_20261014.md)。仓库及 scj 环境已建立，单轨权重和验证产物已存在；不要把下面的迁移初期快照当作当前事实或执行指令。

<details>
<summary>迁移初期历史原文（保留证据，已失效）</summary>

# Progress

## Current phase

Migration/bootstrap on the laboratory A6000 server.

## Completed

- Confirmed the working copy path, SSH identity, Ubuntu kernel, and RTX A6000 hardware identity.
- Confirmed this copy is not currently a Git worktree.
- Confirmed no Slurm commands, project Conda environment, Claude SDK, or MCP Python SDK are currently available in the non-interactive shell.
- Added a dependency-free project MCP worker scaffold for GPT-to-Claude delegation.
- Added project routing/safety rules and A6000-oriented context/handoff/experiment documents.
- Added secret and generated-artifact ignore rules.
- Validated the worker's MCP initialize, tool listing, and non-mutating dry-run responses without calling Claude.
- Confirmed Codex discovers the project-level `claude_worker` server as enabled.
- Confirmed the project secret scan finds no Anthropic key material.

## In progress

- Repository origin and branch strategy are not decided.
- The final Python/CUDA environment has not been migrated or validated.
- Claude API credentials/connectivity are intentionally not live-tested yet.

## Not completed

- No repository has been initialized or cloned.
- No dependency installation has been performed.
- No GPU framework/CUDA tensor test has been run.
- No training command or long-running process has been started.
- Copied paths and scripts have not yet been fully ported from school A100/Slurm assumptions to direct A6000 execution.

## Next phase

1. Decide whether this directory becomes a new repository or is replaced/connected to an existing remote repository.
2. Create or migrate the A6000 Python environment and verify PyTorch/CUDA.
3. Set `ANTHROPIC_API_KEY` in the user/SSH environment (never in the project).
4. Run the worker's non-mutating live API smoke test, then let GPT review the result.
5. Audit training paths/configs and create an A6000 short-smoke-test launcher before any long training.

</details>
