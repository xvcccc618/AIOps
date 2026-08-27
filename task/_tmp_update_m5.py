# -*- coding: utf-8 -*-
"""临时脚本：批量更新模块五复习文档（按行定位替换，保留原换行符）"""
import pathlib

p = pathlib.Path(r"C:\Users\20740\Desktop\Ops\模块五深度复习_去mock化生产化改造.md")
raw = p.read_text(encoding="utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
lines = raw.split(nl)
print(f"total lines: {len(lines)}, line ending: {'CRLF' if nl == chr(13)+chr(10) else 'LF'}")


def find_unique(text, start=0):
    hits = [i for i in range(start, len(lines)) if text in lines[i]]
    assert len(hits) == 1, f"expect 1 hit for {text!r}, got {len(hits)}: {hits}"
    return hits[0]


def replace_range(start_idx, end_idx, new_block):
    """替换 [start_idx, end_idx) 区间为 new_block（list of str）"""
    lines[start_idx:end_idx] = new_block


# ---------- 1. 复习对象行：更新 checkpoint_factory 行数 ----------
i = find_unique("> 复习对象：`settings.py`")
lines[i] = "> 复习对象：`settings.py`(135行) / `observability.py`(111行) / `checkpoint_factory.py`(78行) / `backends.py`(347行) / `tool.py`(269行) / `reranker_service.py`(95行) / `rca_knowledge_base.py`(345行) / `config.ini` / `.env` / `chaos_drill.py` / `verify_real.py` / `test_chain.py`(链路联调)"
print("1. header updated at line", i)

# ---------- 2. 概念 4 正文替换（标题已改，替换标题后到概念5之前） ----------
start = find_unique("### 4. Checkpointer 降级工厂") + 1
end = find_unique("### 5. 真实后端适配层")
concept4 = """
**一句话版**：优先 AsyncRedisSaver 异步持久化（支持 HITL 挂起恢复），Redis 不可用时快速 ping 失败并降级 MemorySaver，主流程不阻断。

**展开版**：
- 先用 `redis.asyncio.from_url().ping()` 异步探测（2 秒超时），避免编译出"名义连接但实际不可用"的 saver
- 探测通过才构造 `AsyncRedisSaver(redis_url=url)` 并 `await saver.asetup()`（首次运行创建 Redis 索引结构，幂等）
- **关键**：主图全部用 `ainvoke` 异步执行，必须用 AsyncRedisSaver——同步版 RedisSaver 未实现 `aget_tuple`，图执行时抛 NotImplementedError（真实联调踩坑，见案例 1）
- `create_checkpointer()` 本身是 async 函数，因此 `build_graph` / `build_multi_agent_graph` 也改为 async，调用方需 await（异步传染性）
- 另保留 `create_checkpointer_sync()` 同步版，仅供故障演练探测等不实际跑图的场景
- 降级只影响断点恢复能力，不影响当次会话执行

**追问预备**：
- "为什么不用同步版 RedisSaver？"→ 同步版没实现异步接口（aget_tuple/aput），配合 ainvoke 执行直接抛 NotImplementedError；这是联调时真实踩到的坑
- "为什么工厂要改成 async？"→ AsyncRedisSaver 的 asetup 是协程，必须在事件循环里 await；工厂一变 async，所有调用方（两个 build 函数）都得跟着变
- "降级后 HITL 怎么办？"→ interrupt 仍能暂停，但进程重启后恢复点丢失，需要用户重新触发
""".strip("\n").split("\n")
replace_range(start, end, concept4)
print("2. concept 4 replaced")

# ---------- 3. 图 2 替换为异步版 ----------
start = find_unique("### 图 2：Checkpointer 降级决策流")
end = find_unique("### 图 3：真实后端适配层调用链")
fig2 = """### 图 2：Checkpointer 降级决策流（异步版）

```
await create_checkpointer()   ← async 工厂（异步传染：build 函数也要 await）
       │
       ▼
┌──────────────────┐
│ 读取 Redis URL    │
│ get_redis_config │
└────────┬─────────┘
         ▼
┌────────────────────────┐    ping 失败（2s 超时）
│ await probe.ping()     │──────────────────────┐
│ redis.asyncio 探测      │                      │
└────────┬───────────────┘                      ▼
         │ ping 成功                     ┌──────────────────┐
         ▼                              │ logger.warning    │
┌────────────────────────┐              │ "Redis 不可用"     │
│ AsyncRedisSaver(       │              └────────┬─────────┘
│   redis_url=url)       │                       ▼
└────────┬───────────────┘              ┌──────────────────┐
         ▼                              │ return            │
┌────────────────────────┐              │ MemorySaver()     │
│ await saver.asetup()   │              │ （无持久化）        │
│ 创建索引结构（幂等）      │              └──────────────────┘
│ 失败 → 走降级            │
└────────┬───────────────┘
         ▼
┌──────────────────────────┐
│ return AsyncRedisSaver    │
│ （持久化 + HITL 恢复）      │
└──────────────────────────┘

要点：同步版 RedisSaver 未实现 aget_tuple，
配合 ainvoke 执行会抛 NotImplementedError → 必须用异步版
```

**默画检查点**：先 ping 后构造、asetup 是协程需 await、异步传染（build 函数也变 async）、降级只影响断点恢复
""".strip("\n").split("\n") + [""]
replace_range(start, end, fig2)
print("3. figure 2 replaced")

# ---------- 4. 题 20 答案升级 ----------
i = find_unique("20. **这套改造里你踩过最深的坑是什么？**")
assert "RedisSaver.from_conn_string" in lines[i + 1]
lines[i + 1] = "    → 参考答案（升级版）：同步版 RedisSaver 配合 ainvoke 异步执行抛 NotImplementedError（同步版未实现 aget_tuple），且报错信息为空极具迷惑性——必须打印异常类型才定位到；换成 AsyncRedisSaver 后又因异步传染把整个 build 函数改成 async。这体现\"真实联调才暴露的隐藏 bug\"。更多案例见第五部分的 8 个真实调试案例。"
print("4. question 20 updated")

# ---------- 5. 插入"真实联调调试案例"章节 + 章节号顺延 ----------
i = find_unique("## 五、自测清单")
cases = """## 五、真实联调调试案例（本次改造的 8 个硬骨头）★区分度核心

> 这一节是面试最有说服力的部分——每个案例都是真实联调中遇到的，按"症状 → 排查 → 解决 → 启示"组织。面试官问"你遇到过最难的问题"时，从这里挑 2-3 个讲。

### 案例 1：同步版 RedisSaver + 异步图执行 = NotImplementedError（框架兼容类）

- **症状**：链路编译通过，但单/多 Agent 一执行就瞬间失败，**报错信息为空**
- **排查**：空报错是最大迷惑点——直接打印异常类型与堆栈，发现是 `NotImplementedError`，位置在 `aget_tuple`；翻 langgraph-checkpoint-redis 包源码，发现同步版 RedisSaver 只实现同步接口，异步版在 `langgraph.checkpoint.redis.aio.AsyncRedisSaver` 子模块
- **解决**：工厂改用 AsyncRedisSaver + `await asetup()`；异步传染——两个 build 函数全部改 async，调用方加 await
- **启示**：空报错先打印异常类型；async/sync API 配对错误是 LangGraph 生态的高频坑

### 案例 2：结构化输出 400 错误四连击（json_mode 兼容，框架兼容类）

- **症状 A**：`Unsupported function ... must be valid JSON schema with top-level 'title' key`——新版 langchain-core 不再接受 `{字段: 类型}` 简写字典
- **症状 B**：改成完整 JSON Schema 后，`This response_format type is unavailable now`——该 LLM 端点不支持默认的 function-calling method
- **症状 C**：改用 json_mode 后，`Prompt must contain the word 'json'`——json_mode 要求 prompt 里必须出现 "json" 字样
- **症状 D**：json_mode 通过后，Pydantic 校验报 24 个字段错误——模型自由发挥的字段名（`step`）与 schema（`step_id`）对不上，**json_mode 不像 function calling 那样强制 schema**
- **解决**：① utils.py 统一两个转换器（简写字典→带 title 的 JSON Schema；Pydantic 模型→解析 `$ref` 的 JSON Schema）；② 全部 12 处 `with_structured_output` 加 `method="json_mode"`；③ planner 提示词补 "JSON" 字样与字段名清单；④ 解析时加字段别名归一化 `_normalize_plan_dict` 兜底
- **启示**：json_mode 只保证"输出是合法 JSON"，不保证"字段符合 schema"——需要提示词约束 + 解析归一化双保险

### 案例 3：Planner-Executor 无限循环（三层根因，图编排类）

- **症状**：流式输出显示 planner→executor 交替无限循环，永不终止
- **根因 1**：planner 提示词不含 "json" → 400 → 返回空计划 → executor 见空计划触发 replan → 回 planner
- **根因 2**：生成相返回 tool_calls 时**没清除上一轮评估相残留的 `next_action="replan"`**，而路由先判 replan 再判 pending_tool_calls → 生成的工具调用永远不执行
- **根因 3**：replan 分支不递增 `replan_count` → "超 3 次兜底"的安全上限永远不触发
- **解决**：① 提示词加 JSON 要求并明确列出全部字段名；② 生成相显式设 `next_action="execute_next"` 覆盖残留信号；③ 空计划分支递增计数、超限转兜底；④ OpsAgentState 补声明缺失字段（未声明字段跨节点会被静默丢弃）
- **启示**：图结构死循环必须"流式按节点输出"定位；LangGraph 未声明的状态字段会被静默丢弃

### 案例 4：工具子图结果字段与主图脱节——静默丢弃（图编排类）

- **症状**：日志反复出现 "Generating tool call for Step 1"，但工具执行日志一条都没有
- **根因**：工具执行子图把结果写入 `tool_messages` 字段，而主图状态**没有声明这个字段** → LangGraph 静默丢弃 → executor 永远收不到工具返回 → 反复重发同一步
- **解决**：子图状态字段对齐主图，工具结果写入 `messages`（经 add_messages reducer 合并回主图）
- **启示**：LangGraph 对未声明字段是**静默丢弃不报错**——"静默数据丢失"比显式报错难查十倍；子图与主图对接必须逐字段核对

### 案例 5：Handoff 评估 KeyError + 角色大小写静默失效（多 Agent 接线类）

- **症状 A**：`[l2_agent] Handoff evaluation failed: 'findings'`——HANDOFF_EVALUATION_PROMPT 有 `{findings}` 占位符但 `format()` 只传了 `agent_role`
- **症状 B**：修完 A 后交接仍从不成功——Prompt 让 LLM 输出大写 "L2_AGENT"，但枚举值是小写 "l2_agent"，`SpecialistRole(raw)` 抛 ValueError 被 except 吞掉 → **交接和路由全部静默失效**
- **解决**：① format 补传 findings 参数；② 新增 `parse_specialist_role()` 健壮解析（大小写不敏感、同时匹配枚举名与枚举值），初始路由/仲裁/Handoff 评估三处统一使用
- **启示**：`except Exception` 全吞是静默失效的最大元凶——捕获异常时要先想"它是不是在掩盖真实 bug"

### 案例 6：反思节点只认英文错误关键词（健壮性细节类）

- **症状**：工具返回中文报错（"执行失败"、"安全拦截"、"未找到工具"），反思节点启发式预筛却判为"正常"，跳过了深度反思
- **解决**：错误关键词列表补充中文项（执行失败/执行超时/安全拦截/未找到工具/熔断/拒绝/不可达等）
- **启示**：中文语境下的关键词列表必须双语覆盖——这类"看起来在工作其实半失效"的 bug 最隐蔽

### 案例 7：MySQL 服务权限迷局（环境基建类）

- **症状**：MySQL9.6 服务停止，直接启动 mysqld 报数据目录 Permission denied（errno 13），当前会话无管理员权限
- **排查链**：查服务状态（sc query）→ 查端口监听（3306 无监听）→ 尝试直接启动（权限拒绝）→ 检查数据目录 ACL → 确认是服务账户权限问题
- **解决**：用户以管理员身份手动启动 MySQL9.6 服务后立即连通（9.6.0，目标库 aiops 存在）
- **启示**：联调前先做"三服务连通性检查"（Redis ping / MySQL SELECT 1 / Milvus 列集合）；**环境问题与代码问题要先分离**，否则会对着环境错误改代码

### 案例 8：输出缓冲导致"假死"误判（工具链类）

- **症状**：测试进程 6 分钟无任何输出，疑似死循环卡死
- **根因**：PowerShell 的 `| Out-String` 会缓冲全部输出，Python 进程的实时日志根本看不到——进程其实在正常跑
- **解决**：改用 `python -u`（无缓冲）运行；再用 `astream` 按节点流式输出定位真实卡点
- **启示**：排查卡死先排除"输出层缓冲"，再怀疑"逻辑层死循环"——否则会把正常进程误杀

### 案例归类速记（面试时按类举例）

| 类别 | 案例 | 一句话标签 |
|---|---|---|
| 框架兼容 | 1、2 | async/sync 配对、json_mode 四连击 |
| 图编排逻辑 | 3、4 | 无限循环三层根因、静默字段丢失 |
| 多 Agent 接线 | 5 | KeyError + 大小写静默失效 |
| 健壮性细节 | 6 | 中文关键词缺失 |
| 环境与工具链 | 7、8 | 服务权限迷局、输出缓冲假死 |

---

""".split("\n")
# 原"五、自测清单"顺延为"六"，后续章节同步顺延
lines[i] = "## 六、自测清单"
lines[i:i] = cases
print("5. cases section inserted before self-test")

i = find_unique("## 六、模块衔接")
lines[i] = "## 七、模块衔接"
i = find_unique("## 七、30 秒项目讲解话术")
lines[i] = "## 八、30 秒项目讲解话术（去 mock 化部分）"
print("6. section numbers shifted")

# ---------- 7. 模块衔接补充异步说明 ----------
i = find_unique("**与模块一**：checkpoint_factory 替换了")
lines[i] = "- **与模块一**：checkpoint_factory（异步版）替换了 route.py 和 multi_agent_graph.py 的 checkpointer 创建逻辑，两个 build 函数因此改为 async；observability 的 setup_logging 在 main.py 入口调用"
i = find_unique("**与模块四**：tool.py 的 TOOL_MAP")
lines[i] = "- **与模块四**：tool.py 的 TOOL_MAP 和 TOOL_TIMEOUTS 被 specialist_agents.py 的 ReAct 循环使用；安全校验层保护所有工具调用；本次改造中 L1 占位节点接线为真实子图（见模块四概念 12）"
print("7. module links updated")

# ---------- 8. 自测清单补充案例项 ----------
i = find_unique("### 口述类")
# 找到口述类最后一个清单项（"能讲一个"真实接入才暴露的 bug""）
j = find_unique('能讲一个"真实接入才暴露的 bug"')
extra = [
    '- [ ] 能完整讲出案例 1 的排查链："空报错 → NotImplementedError → aget_tuple → AsyncRedisSaver → 异步传染"',
    "- [ ] 能讲出案例 2 json_mode 兼容的四个症状（简写字典→function calling→prompt 含 json→字段归一化）",
    "- [ ] 能讲出案例 3 无限循环的三层根因（提示词缺 json / 残留 replan 信号 / 计数不递增）",
    "- [ ] 能讲出案例 4 \"LangGraph 静默丢弃未声明字段\"这个坑",
    "- [ ] 8 个案例能按五大类别归类并各举一个",
]
lines[j + 1:j + 1] = extra
print("8. self-test items added")

# ---------- 9. 30 秒话术升级 ----------
i = find_unique('"这个项目我做了完整的去 mock 化生产化改造')
lines[i] = '"这个项目我做了完整的去 mock 化生产化改造：配置上敏感信息收敛到 .env 并进 .gitignore，非敏感项留 config.ini；可观测性用 structlog 桥接现有 logging 零侵入接入，LLM 指标通过 callback 自动采集；工具层接了 K8s/MySQL/Prometheus 三个真实后端，全部带优雅降级——集群不可达返回结构化提示而不是崩溃；checkpointer 从 MemorySaver 换成 AsyncRedisSaver 并做了 ping 探测降级。联调阶段我解决了一批真实问题：同步版 checkpointer 与异步图执行不兼容、结构化输出的 json_mode 四连击兼容、planner-executor 无限循环的三层根因、工具子图结果字段被静默丢弃——这套改造让我理解了\'部分可用\'比\'全有全无\'更重要的生产设计哲学，也理解了\'真实联调是暴露隐藏断点的唯一途径\'。"'
print("9. 30s pitch updated")

p.write_text(nl.join(lines), encoding="utf-8")
print("DONE, new total lines:", len(lines))
