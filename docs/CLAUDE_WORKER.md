# Claude worker：配置沿革与当前使用边界

最近补记：2026-10-23（UTC+08:00，北京时间）。本文件持续保留早期工具配置、路由做法，以及后来对授权和可用性声明的修正。历史正文已恢复在原文件内，后续进展按日期追加；归档仅作备份，无需离开本文件才能阅读旧记录。

阅读顺序：[此前累计正文](#history-records) → [2026-10-23阶段更新](#update-20261023) → [本次历史恢复及生效约定](#history-repair)。旧段落的“当前”“下一步”、命令和PID按其记录时点理解；较晚的更正与实际完成证据在后文，旧命令不因此重新执行。

截至本次修复：Pilot8系列及Complex12的C/U/S各300更新、五方法60条81帧已完成。本次仅修复文档；[实验入口](../experiments/README.md)和[文档索引](../docs/DOCUMENTATION_INDEX.md)仍可直接使用。

当前授权边界：旧文中的关键词自动路由已被后续规则替代；普通“执行/修改”不会自动调用外部worker，本次未委派或调用API。

<a id="history-records"></a>

## 一、此前累计正文：此次整理前长期保留的内容

以下完整保留原有记录顺序、参数、命令、过程和当时结论，只调整标题层级。原记录中的未完成状态与后来完成状态共同构成历史，不能用后者替换前者。

<!-- RESTORED_HISTORY_BEGIN -->
### Claude Worker Setup and Use

> Current project state and operational rules are in [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md), [AGENTS.md](../AGENTS.md), and [SESSION_HANDOFF.md](../SESSION_HANDOFF.md). Migration-era Git/Python statements below are historical. This maintenance session did not invoke Claude or test its API connectivity.

#### What is already configured

Codex loads the project-level `claude_worker` MCP server from `.codex/config.toml`. The current local configuration starts it with `python3` resolved through PATH; it is separate from the scj training interpreter. Its only user-facing tool is `claude_execute`.

The worker uses the existing SSH proxy environment for outbound HTTPS. A6000 compute and Claude API access are separate: the GPU host does not need an independent public-network route as long as the SSH reverse proxy remains available.

#### One-time API-key setup

When the Claude API key is ready, open the SSH terminal in this workspace and run:

```bash
bash scripts/configure_claude_key.sh
```

The script reads the key without echoing it, stores it at `~/.config/claude-worker/env` with mode `600`, and adds an early source line to `~/.bashrc`. The key is outside the project and outside Git. Reconnect the Codex SSH workspace once afterward so the app process inherits the variable.

To choose a different currently available Claude model, add an environment variable named `CLAUDE_WORKER_MODEL`. The scaffold currently defaults to `claude-sonnet-5`.

#### Routing behavior

- “分析、规划、检查、审查、评估、比较、先不要执行”：GPT handles the request itself and does not call Claude.
- “可以，执行 / 开始做 / 按刚才方案修改”：GPT prepares a bounded handoff, calls `claude_execute`, then independently reviews the returned changes and evidence.
- Training remains blocked unless the current user request explicitly approves training and GPT sets both command and training permission for that single delegation.

#### Limitations

- Natural-language routing is enforced by project instructions and the tool description, not by a hidden deterministic keyword router. GPT remains responsible for interpreting intent safely.
- The worker is intentionally small. It provides bounded file inspection/editing and non-interactive command execution, not a second persistent chat UI.
- The first real API request and the final model name cannot be validated before a working API key is available.
- Git is available; use the actual status and diff for review. API connectivity/model availability require a separately authorized live test.
<!-- RESTORED_HISTORY_END -->

<a id="update-20261023"></a>

## 二、2026-10-23：阶段更新与当前资料补全

下面继续保留357b09f阶段补充的结果、参数和操作说明。该阶段曾把历史移到归档、部分原文页改成导航；这一文档组织做法已由第三节的用户更正取代，实验事实和既有检查结果保留。

<!-- RETAINED_UPDATE_BEGIN -->
### 可选Claude worker：本地实现与使用边界

更新：2026-10-23（UTC+09:00）。本页依据项目现有 [claude_worker_mcp.py](../scripts/claude_worker_mcp.py) 和配置脚本说明。本次仅维护Markdown，**没有调用外部worker、发送项目内容给API或测试模型可用性**。

#### 1. 本地配置与环境

项目MCP配置保留claude_worker入口，使用PATH中的python3；这与训练使用的/home/shi/miniconda3/envs/scj/bin/python是不同选择。配置存在只证明有本地入口，不能证明远端API连接、账号权限或模型服务可用。

| 环境变量 | 本地代码用途 |
| --- | --- |
| CLAUDE_WORKER_ROOT | 限定工作根，缺省为启动时cwd |
| CLAUDE_WORKER_MODEL | 请求模型名，缺省字符串claude-sonnet-5 |
| ANTHROPIC_BASE_URL | API根地址，缺省为api.anthropic.com |
| ANTHROPIC_API_KEY | 从环境读取的凭据 |

上述默认模型名仅是代码里的字符串，本次没有在线确认它是否有效。网络代理与SSH转发可用性也未在此次文档任务测试，不沿用历史连接状态作当前保证。

#### 2. 何时可以使用

只有用户或适用工作规则明确要求委派、调用worker或并行代理工作时才调用。普通“执行”“开始修改”“更新文档”不是自动转交外部模型的指令；当前任务由本代理完成。

委派时应明确任务、允许路径、输入范围、是否可编辑、命令权限及验收条件。主任务仍需检查返回差异和实际证据，不能把worker的完成声明直接当作测试或训练成功。

#### 3. 代码中的限制

- allowed_paths与解析后的实际路径限定文件访问，避免通过符号链接越出允许范围。
- data、models、checkpoints、cache、outputs是文件编辑工具的只读顶层目录。
- allow_commands默认false；allow_training默认false，启用训练还要求命令权限。
- 命令过滤是实现中的约束机制，不等于操作系统沙箱，也不保证任意shell都无法绕过文件工具限制。
- dry_run只检查本地请求准备信息，不调用API，不能验证账号或模型可用性。

当前项目训练/推理授权仍按 [AGENTS](../AGENTS.md) 和 [TRAINING](../TRAINING.md)执行；委派权限不能隐含授权追加GPU预算、覆盖成果或改动冻结配置。

#### 4. 凭据设置脚本的实际行为

[configure_claude_key.sh](../scripts/configure_claude_key.sh)在用户明确配置工具时可用：无回显读取key，写入用户目录~/.config/claude-worker/env并设权限600，同时在~/.bashrc加入加载语句。它会修改用户环境，不是纯检查，本次没有执行。

凭据不写进仓库、文档、日志或结果报告，不在工具输出中打印。模型与API可用性需另行明确的在线测试；本页不提供未经验证的可用模型建议。

#### 5. 历史说明

旧文档曾按“分析/执行”等用词自动路由到worker，该描述已停止作为当前操作规则。原文保留在 [更新前快照](history/20261023_before_md_refresh/docs/CLAUDE_WORKER.md)，仅用于追溯，不覆盖当前会话授权边界。

项目状态看 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，目录和文档范围看 [PROJECT_STRUCTURE](PROJECT_STRUCTURE.md)。此工具与VACE完成状态没有因果关系，不能因为它存在而启动额外任务。
<!-- RETAINED_UPDATE_END -->

<a id="history-repair"></a>

## 三、2026-10-23：恢复原文件内的持续历史

用户明确要求长期MD保留“从以前到现在”的更新。先前仅保留当前摘要、让读者另找归档的处理不符合要求；本次将上述旧正文直接恢复到原文件，同时保留后续更新，没有回退实验结果或把完成状态改回执行中。

以后在本文件按日期追加阶段进展、失败与修复、参数变更、结论和核查范围。需要纠正旧结论时，说明原判断、后来的证据及新结论；原记录继续保留。可以在顶部维护简短状态和阅读索引，不能用摘要或归档链接替换历史正文。

本次修复依据原文件备份及Git记录恢复，没有补造新的运行经历；原数据、权重、视频、指标、冻结报告和现有非Markdown工作保持原样。此次检查与恢复范围见[文档索引](../docs/DOCUMENTATION_INDEX.md)中的后续补记。
