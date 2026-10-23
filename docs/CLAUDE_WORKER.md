# 可选Claude worker：本地实现与使用边界

更新：2026-10-23（UTC+09:00）。本页依据项目现有 [claude_worker_mcp.py](../scripts/claude_worker_mcp.py) 和配置脚本说明。本次仅维护Markdown，**没有调用外部worker、发送项目内容给API或测试模型可用性**。

## 1. 本地配置与环境

项目MCP配置保留claude_worker入口，使用PATH中的python3；这与训练使用的/home/shi/miniconda3/envs/scj/bin/python是不同选择。配置存在只证明有本地入口，不能证明远端API连接、账号权限或模型服务可用。

| 环境变量 | 本地代码用途 |
| --- | --- |
| CLAUDE_WORKER_ROOT | 限定工作根，缺省为启动时cwd |
| CLAUDE_WORKER_MODEL | 请求模型名，缺省字符串claude-sonnet-5 |
| ANTHROPIC_BASE_URL | API根地址，缺省为api.anthropic.com |
| ANTHROPIC_API_KEY | 从环境读取的凭据 |

上述默认模型名仅是代码里的字符串，本次没有在线确认它是否有效。网络代理与SSH转发可用性也未在此次文档任务测试，不沿用历史连接状态作当前保证。

## 2. 何时可以使用

只有用户或适用工作规则明确要求委派、调用worker或并行代理工作时才调用。普通“执行”“开始修改”“更新文档”不是自动转交外部模型的指令；当前任务由本代理完成。

委派时应明确任务、允许路径、输入范围、是否可编辑、命令权限及验收条件。主任务仍需检查返回差异和实际证据，不能把worker的完成声明直接当作测试或训练成功。

## 3. 代码中的限制

- allowed_paths与解析后的实际路径限定文件访问，避免通过符号链接越出允许范围。
- data、models、checkpoints、cache、outputs是文件编辑工具的只读顶层目录。
- allow_commands默认false；allow_training默认false，启用训练还要求命令权限。
- 命令过滤是实现中的约束机制，不等于操作系统沙箱，也不保证任意shell都无法绕过文件工具限制。
- dry_run只检查本地请求准备信息，不调用API，不能验证账号或模型可用性。

当前项目训练/推理授权仍按 [AGENTS](../AGENTS.md) 和 [TRAINING](../TRAINING.md)执行；委派权限不能隐含授权追加GPU预算、覆盖成果或改动冻结配置。

## 4. 凭据设置脚本的实际行为

[configure_claude_key.sh](../scripts/configure_claude_key.sh)在用户明确配置工具时可用：无回显读取key，写入用户目录~/.config/claude-worker/env并设权限600，同时在~/.bashrc加入加载语句。它会修改用户环境，不是纯检查，本次没有执行。

凭据不写进仓库、文档、日志或结果报告，不在工具输出中打印。模型与API可用性需另行明确的在线测试；本页不提供未经验证的可用模型建议。

## 5. 历史说明

旧文档曾按“分析/执行”等用词自动路由到worker，该描述已停止作为当前操作规则。原文保留在 [更新前快照](history/20261023_before_md_refresh/docs/CLAUDE_WORKER.md)，仅用于追溯，不覆盖当前会话授权边界。

项目状态看 [PROJECT_CONTEXT](../PROJECT_CONTEXT.md)，目录和文档范围看 [PROJECT_STRUCTURE](PROJECT_STRUCTURE.md)。此工具与VACE完成状态没有因果关系，不能因为它存在而启动额外任务。
