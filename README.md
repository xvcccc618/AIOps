# IT-Ops Agent

企业级智能 IT 运维与故障排查 Agent，基于 **LangGraph** 构建。支持单 Agent SOP 排查与多 Agent 专家协作两种模式，覆盖 RAG 检索、根因分析（RCA）、知识反哺、幻觉治理、权限审计与评测闭环。

## 量化评测结果

内置自动化评测套件（`task/test_evaluation_suite.py`），共 **43 项测试、7 个维度**，整体通过率 **93.0%（40/43）**，自动产出 Markdown + JSON 双格式报告。

| 维度 | 结果 | 关键指标 |
|---|---|---|
| RAG 检索质量 | 6/9 | 空检索率 **0.0%**（阈值 ≤50%）；平均检索耗时 **4.3s**（阈值 ≤5s）；6 条标准故障 Query 关键词命中率 50%，3 条未命中均因知识库案例覆盖不足 |
| 诊断准确性 | 5/5 | RCA 证据链强约束（每条根因绑定证据）；置信度校准 0.85；Action Items SMART 合规率 **100%** |
| 执行效率 | 5/5 | 工具可调用率 **100%**（6/6）；重规划 ≥3 次自动转兜底；多 Agent 死循环（L1→L2→L1）成功检测；Handoff 上限 3 次 |
| 成本控制 | 5/5 | 按 P0~P4 故障等级分级 token 预算；启发式预筛过滤正常步骤的 LLM 反思调用；500 行日志压缩至 4000 token 预算内（实测 3537 字符） |
| 系统韧性 | 8/8 | Redis 不可用自动降级内存检查点；工具超时配置覆盖率 **100%**（10~15s）；无证据数值声明幻觉检出率 100%（3 处证据缺口） |
| 安全合规 | 8/8 | 注入攻击拦截率 **4/4（100%）**；高危操作两级审批；密钥零明文（只走 .env） |
| 评测系统自检 | 3/3 | RAGAS 四指标计算、Agent 行为追踪、聚合报告全部正确 |

```bash
cd task
python test_evaluation_suite.py   # 运行评测，输出 test_report.md / test_report.json
```

## 架构概览

```
用户报障 → router(意图识别)
            ├─ RAG问答: graph_expansion → rag_retrieval → generator
            ├─ 故障排查: topology_query → planner → executor ⇄ tool_executor
            │                                    → reflect_and_adjust(反思)
            │                                    → critic → rca_generator → distributor
            ├─ 闲聊 / 追问澄清 → 直接应答
            └─ 多Agent模式: supervisor → L1/L2/DBA 专家子图（P2P Handoff + 仲裁）
```

## 核心特性

| 模块 | 说明 |
|---|---|
| **图编排** | LangGraph StateGraph，15 节点主图 + 工具执行子图，Redis Checkpoint 持久化，Redis 不可用自动降级内存检查点 |
| **RAG 检索** | 双路召回（纯语义 + 拓扑图引导）→ RRF 融合 → BGE-Reranker 精排 → 父子块智能预算装配，向量库为 Milvus（支持增量写入与标量过滤） |
| **Plan-Execute-Reflect** | LLM 生成 SOP 排查计划，执行后反思，失败路径黑名单防重复规划，三次重置失败升级人工介入 |
| **多 Agent 协作** | Supervisor + P2P Handoff 混合模式，已排除假设跨 Agent 传递防重复排查，交接计数 + 死循环检测 + 强制仲裁 |
| **知识闭环** | RCA 报告自动反哺向量库与拓扑图（三重防护防污染），系统越用越准 |
| **可靠性** | 高危操作两级人工审批（interrupt）、熔断器、指数退避重试、HITL 超时看门狗（Redis TTL） |
| **安全合规** | RBAC 角色权限、工具参数注入防护、只读 SQL 三层校验、RAG 方案高危操作审计 |
| **可观测性** | structlog 结构化日志、LLM 调用 token/耗时/成本采集、RAGAS 评测指标 |

## 技术栈

- **编排**：LangGraph / LangChain
- **LLM**：DeepSeek（OpenAI 兼容接口，可替换）
- **向量库**：Milvus 2.4（database: itcast, collection: edurag_final, 1024 维 COSINE）
- **Embedding / Rerank**：BAAI/bge-m3、BAAI/bge-reranker-v2-m3（SiliconFlow 远程 API，无本地模型依赖，可直接容器化）
- **Checkpoint / 缓存**：Redis（checkpointer、HITL 审批超时看门狗）
- **拓扑图**：NetworkX（时间感知边）

## 目录结构

```
task/
├── main.py                  # 入口：单Agent/多Agent/交互模式
├── route.py                 # 主图构建（15节点）与 planner/executor
├── multi_agent_graph.py     # 多Agent协作图（Supervisor + P2P）
├── supervisor.py            # 路由/仲裁/汇总
├── specialist_agents.py     # L2/DBA 专家子图与 ReAct 循环
├── retrieve.py              # RAG 五步检索流水线
├── rca_knowledge_base.py    # Milvus 向量库 + 父子块索引
├── reranker_service.py      # BGE 远程重排
├── rca_generator.py         # RCA 报告生成（证据链 + SMART 约束）
├── rca_ingestion.py         # 知识反哺（向量库 + 拓扑图双写）
├── reflection_node.py       # 反思节点（启发式预筛 + 防甩锅交叉验证）
├── token_budget_manager.py  # 按严重等级分级的 token 预算
├── rbac.py / audit.py       # 权限控制与审计
├── hallucination_guard.py   # 幻觉防护与会话超时
├── mcp_server.py / mcp_client.py  # MCP 工具协议层
├── tool.py                  # K8s/DB 工具集（含参数安全校验）
├── settings.py              # 集中配置（密钥只来自 .env）
├── observability.py         # structlog + LLM 指标采集
├── checkpoint_factory.py    # Redis 检查点（含降级）
├── evaluation.py            # RAGAS 指标计算与 Agent 行为追踪
├── test_evaluation_suite.py # 量化评测套件（7 维度 43 项测试）
├── chaos_drill.py           # 故障演练脚本
├── verify_real.py           # 真实端点冒烟验证
└── config.ini               # 非敏感配置（Redis/Milvus 地址）
```

## 快速开始

### 1. 启动基础设施

```bash
# Redis + Milvus（见 docker-compose.yml，含 etcd/minio/redis）
docker compose up -d
```

### 2. 安装依赖

```bash
pip install -r task/requirements.txt
```

### 3. 配置密钥

```bash
cp task/.env.example task/.env
# 编辑 .env，填入 DeepSeek 与 SiliconFlow 的 API Key
```

### 4. 运行

```bash
cd task
python main.py --query "订单服务 Pod 频繁重启 CrashLoopBackOff OOM"
# 交互模式
python main.py
```

### 5. 验证与评测

```bash
python verify_real.py             # 真实端点链路验证（LLM/Embedding/Milvus/Rerank/RAG）
python chaos_drill.py             # 故障演练（断依赖降级验证）
python test_evaluation_suite.py   # 量化评测（7 维度 43 项，输出双格式报告）
```
