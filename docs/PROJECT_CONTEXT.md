# Project Context

## Stable goal

Fine-tune and evaluate Wan2.1-T2V-1.3B with LoRA so that the model learns the appearance and complete rotating views of general building models.

## Current execution environment

- Current working copy: `/mnt/windowsE/chuanjun/building_wan`
- Current SSH host/user: `shi@vccip`
- Target GPU: NVIDIA RTX A6000 (hardware detected through PCI/driver metadata)
- Operating system: Ubuntu Linux 6.8
- Scheduler: no Slurm commands detected on this host; do not use copied school-cluster `sbatch` instructions here.
- Python environment: not migrated yet. The only currently confirmed interpreter is `/usr/bin/python3`.
- Internet/API path: the SSH session inherits the existing reverse-proxy environment. Claude API connectivity is not considered verified until a safe API smoke test succeeds.

The root-level `PROJECT_CONTEXT.md`, `TRAINING.md`, `SESSION_HANDOFF.md`, and `EXPERIMENTS.md` were copied from the former school A100/Slurm environment. They are historical migration evidence, not instructions for the A6000 host.

## Main modules

- `code/Python_3D_Scanner`: project-owned collection, preparation, training, validation, and experiment logic.
- `code/DiffSynth-Studio`: third-party Wan training/inference engine; modify only when the project layer cannot solve the issue.
- `data/`, `models/`, `cache/`, `checkpoints/`, `outputs/`: large or generated assets; read-only by default.

## Agent architecture

- GPT Codex is the sole main agent, planner, reviewer, risk owner, and final decision-maker.
- Claude API is an execution worker invoked only after explicit user approval.
- The project-level MCP server `claude_worker` exposes one delegation tool to Codex and runs inside the SSH workspace.
- Claude receives bounded project file and command tools. It cannot write to protected artifact directories through its file tool, cannot run destructive commands, and cannot train unless GPT explicitly enables training for that delegation.
- Every Claude execution result returns to GPT for diff/config/log/result review.

## Architecture decisions

1. Use a small project-local stdio MCP server instead of a second chat window or a large orchestration framework.
2. Use only Python standard-library code at bootstrap time, so the integration can be validated before the project Conda environment exists.
3. Read credentials only from environment variables. Never store API keys in the repository, handoff, or logs.
4. Preserve copied school-cluster experiment records as history while creating new A6000-specific progress and experiment records here.
5. Until Git is created or cloned, report Git state as unavailable; never fabricate a branch, commit, or diff.
