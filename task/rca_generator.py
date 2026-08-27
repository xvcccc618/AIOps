import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from bone import OpsAgentState, AgentStatus
from rca_schema import RCAResult
from utils import pydantic_model_to_json_schema

logger = logging.getLogger("RCAGeneratorNode")

# Few-Shot Examples for SMART Action Items
SMART_EXAMPLES = """
【Action Items 编写规范 - SMART 原则】
 错误示例 (模糊): "优化数据库性能", "加强监控", "注意代码质量"
 正确示例 (SMART): 
   - "为 order_db 表的 user_id 字段添加联合索引 (user_id, create_time)，预计降低 P99 延迟至 50ms 以内。Owner: DBA-Team, Deadline: 2025-02-01"
   - "在 Payment-Service 中集成 Prometheus 自定义指标，监控 Druid 连接池等待线程数，当 >10 时触发 P1 告警。Owner: SRE-Backend, Deadline: 2025-02-15"

【Root Cause 证据链要求】
每一个根因结论必须绑定至少一个证据。
例如：
- 结论: "Druid 连接池耗尽"
- 证据: 
  1. Log: "Get connection timeout, wait millis 60000" from pod payment-service-abc
  2. Metric: Druid ActiveConnections reached max (50/50) at 10:05 AM
"""

RCA_GENERATION_PROMPT = f"""你是一名资深 SRE 专家，负责撰写最终的故障根因分析 (RCA) 报告。

{SMART_EXAMPLES}

【输入信息】
1. **原始故障描述**: {{query}}
2. **排查计划与执行历史**: 
   {{execution_history}}
3. **工具调用关键结果**: 
   {{tool_results_summary}}
4. **历史案例参考 (RAG)**: 
   {{rag_context}}
5. **专家裁决意见 (Critic)**: 
   {{critic_decision}}
6. **拓扑上下文**: 
   {{topology_context}}

【撰写要求】
1. **严格基于事实 (Evidence-Based)**：
   - 在 `root_cause_analysis` 中，每得出一个结论，必须在 `evidence_chain` 中列出支撑该结论的具体日志片段、监控指标或工具输出。
   - **无证据不结论**。如果缺乏直接证据，请说明“推测性结论，需进一步验证”。
2. **客观中立**：
   - 避免使用“甩锅”语言。聚焦于系统机制失效，而非个人失误。
   - 如果多个团队涉及，明确指出交互界面的问题（如 API 超时设置不一致）。
3. **SMART Action Items**：
   - 严禁生成“优化代码”、“加强监控”等模糊建议。
   - 必须包含：具体动作、量化目标、明确 Owner、明确 Deadline。
4. **数据缺失处理**：
   - 如果输入中未提供具体的“受损金额”、“具体用户数”，在 `impact_assessment` 中必须写出：“数据缺失，需人工补充”。

【输出格式】
请输出符合 RCAResult Schema 的 JSON 对象。
"""


async def rca_generator_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    RCA 生成节点：
    1. 聚合 State 中的所有关键信息。
    2. 调用 LLM 生成结构化 RCA 报告 (带证据链和 SMART 约束)。
    3. 初步校验。
    """
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "rca_generator", "error": "LLM missing"}]}

    # 1. 聚合上下文
    query = state.get("query", "")

    execution_history = state.get("execution_history", [])
    history_str = "\n".join([
        f"- Step {i + 1} [{h['step_info']['action']}]: {h['result_summary']}"
        for i, h in enumerate(execution_history)
    ]) if execution_history else "无详细执行记录。"

    messages = state.get("messages", [])
    tool_results = []
    for msg in reversed(messages):
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'tool':
            # 截取关键部分，保留 Tool Call ID 以便追溯
            tool_results.append(f"[Tool ID: {msg.tool_call_id}] {msg.content[:500]}")
            if len(tool_results) >= 5: break
    tool_results_str = "\n---\n".join(tool_results) if tool_results else "无额外工具结果。"

    rag_context = "\n".join(state.get("retrieved_context", [])) if state.get("retrieved_context") else "无相关历史案例。"

    critic_decision = state.get("critic_decision", {})
    critic_str = ""
    if isinstance(critic_decision, dict):
        critic_str = f"Verdict: {critic_decision.get('final_verdict', 'N/A')}\nReasoning: {critic_decision.get('reasoning', 'N/A')}"
    else:
        critic_str = str(critic_decision)

    topology_context = state.get("topology_context", "无拓扑信息。")

    # 2. 构建 Prompt
    prompt_content = RCA_GENERATION_PROMPT.format(
        query=query,
        execution_history=history_str,
        tool_results_summary=tool_results_str,
        rag_context=rag_context,
        critic_decision=critic_str,
        topology_context=topology_context
    )

    # 3. 调用 LLM (Structured Output) —— 使用 json_mode 兼容当前端点
    try:
        structured_llm = llm.with_structured_output(
            pydantic_model_to_json_schema(RCAResult), method="json_mode"
        )

        raw = await structured_llm.ainvoke([
            SystemMessage(content="你是一个严谨、客观、注重证据的 SRE 故障分析专家。"),
            HumanMessage(content=prompt_content)
        ])

        # json_mode 返回 dict，校验回 Pydantic 模型
        response = RCAResult(**raw)

        logger.info(f"[RCAGenerator] Successfully generated RCA report. Confidence: {response.confidence_score}")
        logger.info(f"[RCAGenerator] Evidence Chain Length: {len(response.evidence_chain)}")

        # 4. 返回结果
        return {
            "rca_report_json": response.model_dump(),
            "rca_distribution_status": "PENDING",
            "status": AgentStatus.SUCCESS,
            "final_answer": f"RCA 报告已生成。\n\n**故障简述**: {response.incident_summary[:100]}...\n\n请查看完整报告或等待自动分发。"
        }

    except Exception as e:
        logger.error(f"[RCAGenerator] Failed to generate structured RCA: {e}", exc_info=True)
        return {
            "status": AgentStatus.FAILED,
            "error_log": [{"node": "rca_generator", "error": str(e)}],
            "final_answer": "自动生成 RCA 报告失败，请稍后重试或人工介入。"
        }
