import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from bone import (
    MultiAgentSupervisorState, AgentStatus,
    SpecialistRole, AgentHandoff, parse_specialist_role
)

logger = logging.getLogger("SupervisorNode")

from utils import to_structured_output_schema

MAX_HANDOFF_COUNT = 3

SUPERVISOR_ROUTING_PROMPT = """你是一个高级 SRE 故障排查指挥官 (Supervisor)。你的任务是分析当前的故障现象，决定将任务分配给哪位 Specialist Agent。
【可选 Specialist】
1. L1_AGENT: 初步信息收集、查日志、看监控。
2. L2_AGENT: 深度代码分析、链路追踪、复杂逻辑。
3. DBA_AGENT: SQL 慢查询、锁竞争、连接池诊断。

【当前故障信息】
用户 Query: {query}
提取的实体: {entities}
拓扑上下文: {topology_context}

请以 JSON 格式输出：
{{
    "selected_agent": "L1_AGENT" | "L2_AGENT" | "DBA_AGENT",
    "reasoning": "选择该 Agent 的理由",
    "task_description": "给该 Agent 的具体任务描述和排查重点"
}}
"""


ARBITRATION_PROMPT = """你是一个高级 SRE 故障排查仲裁官。当前排查陷入了僵局，多个专家互相推诿或达到了最大交接次数。
【各方发现】
{findings_summary}
【交接历史】
{handoff_history}
【全局已排除项】
{excluded_hypotheses}

【任务】
请综合所有信息，做出最终裁决：
1. 指出真正的根因方向。
2. 指定最终负责确认该根因的 Agent (L1/L2/DBA)。如果认为信息已足够，可指定 "NONE" 直接输出结论。
3. 给出最终结论或给最终 Agent 的明确指令。

输出 JSON:
{{
    "verdict": "最终裁决结论",
    "final_agent": "L1_AGENT" | "L2_AGENT" | "DBA_AGENT" | "NONE",
    "final_task": "给最终 Agent 的任务，或直接输出的最终结论"
}}
"""


def detect_deadlock(history: List[AgentHandoff]) -> bool:
    if len(history) < 2:
        return False
    last = history[-1]
    prev = history[-2]
    return last.from_agent == prev.to_agent and last.to_agent == prev.from_agent


async def supervisor_node(state: MultiAgentSupervisorState, config: RunnableConfig) -> dict:
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "supervisor", "error": "LLM missing"}]}

    query = state.get("query", "")
    current_specialist = state.get("current_specialist")
    handoff_count = state.get("handoff_count", 0)
    handoff_history = state.get("handoff_history", [])
    is_arbitrating = state.get("is_arbitrating", False)

    # 场景 1：初始路由
    if not current_specialist or current_specialist == SpecialistRole.SUPERVISOR:
        logger.info("[Supervisor] Performing initial routing...")
        entities = state.get("extracted_entities", {})
        topology_ctx = state.get("topology_context", "无")

        try:
            structured_llm = llm.with_structured_output(to_structured_output_schema({
                "selected_agent": str, "reasoning": str, "task_description": str
            }), method="json_mode")
            response = await structured_llm.ainvoke([
                SystemMessage(content=SUPERVISOR_ROUTING_PROMPT.format(
                    query=query, entities=entities, topology_context=topology_ctx
                )),
                HumanMessage(content="请进行初始路由决策。")
            ])
            # 健壮解析角色（LLM 可能输出 "L1_AGENT" 大写枚举名，解析失败兜底 L1）
            selected_role = parse_specialist_role(response.get("selected_agent")) or SpecialistRole.L1_AGENT
            return {
                "current_specialist": selected_role,
                "messages": [AIMessage(content=f"[Supervisor] 路由至 {selected_role.value}")],
                "status": AgentStatus.RUNNING,
                "_task_for_specialist": response.get("task_description", "初步收集信息。")
            }
        except Exception as e:
            logger.error(f"[Supervisor] Routing failed: {e}")
            return {"current_specialist": SpecialistRole.L1_AGENT, "status": AgentStatus.RUNNING,
                    "_task_for_specialist": "初步收集信息。"}

    # 场景 2:仲裁模式 (Handoff 超限或检测到死循环)
    if is_arbitrating or handoff_count >= MAX_HANDOFF_COUNT or detect_deadlock(handoff_history):
        logger.warning(
            f"[Supervisor] Triggering Arbitration! Count: {handoff_count}, Deadlock: {detect_deadlock(handoff_history)}")

        findings_summary = "\n---\n".join([f"[{k}]: {v}" for k, v in state.get("specialist_findings", {}).items()])
        history_str = "\n".join([f"{h.from_agent.value} -> {h.to_agent.value}: {h.reason}" for h in handoff_history])
        excluded_str = "\n".join(state.get("_excluded_hypotheses", [])) or "无"

        try:
            structured_llm = llm.with_structured_output(to_structured_output_schema({
                "verdict": str, "final_agent": str, "final_task": str
            }), method="json_mode")
            response = await structured_llm.ainvoke([
                SystemMessage(content=ARBITRATION_PROMPT.format(
                    findings_summary=findings_summary,
                    handoff_history=history_str,
                    excluded_hypotheses=excluded_str
                )),
                HumanMessage(content="请进行最终仲裁。")
            ])

            final_role = parse_specialist_role(response.get("final_agent"))
            if final_role is None:
                # NONE 或解析失败：直接输出结论
                return {
                    "status": AgentStatus.SUCCESS,
                    "final_answer": f"【仲裁结论】{response.get('verdict', '')}\n\n{response.get('final_task', '')}"
                }
            else:
                # 强制指定最终 Agent
                return {
                    "current_specialist": final_role,
                    "status": AgentStatus.RUNNING,
                    "_task_for_specialist": f"【仲裁强制指令】{response.get('verdict', '')}\n任务: {response.get('final_task', '')}",
                    "is_arbitrating": False  # 重置仲裁状态
                }
        except Exception as e:
            logger.error(f"[Supervisor] Arbitration failed: {e}")
            return {"status": AgentStatus.SUCCESS, "final_answer": "仲裁失败，请人工介入。"}

    # 场景 3：正常结果汇总 (P2P 结束或 Specialist 主动返回 Supervisor)
    specialist_value = current_specialist.value if isinstance(current_specialist, SpecialistRole) else str(current_specialist)
    logger.info(f"[Supervisor] Evaluating findings from {specialist_value}...")
    current_findings = state.get("specialist_findings", {}).get(specialist_value, "无")

    # 简单汇总逻辑：如果 Specialist 返回 SUCCESS，则结束
    if state.get("status") == AgentStatus.SUCCESS:
        return {
            "status": AgentStatus.SUCCESS,
            "final_answer": f"【{specialist_value} 排查结论】\n{current_findings}"
        }

    # 如果失败，尝试重新路由
    return {"status": AgentStatus.FAILED, "final_answer": "排查异常终止。"}


def route_after_supervisor(state: MultiAgentSupervisorState) -> str:
    status = state.get("status")
    if status == AgentStatus.SUCCESS or status == AgentStatus.FAILED:
        return "end"

    target = state.get("current_specialist")
    if target == SpecialistRole.L1_AGENT:
        return "l1_subgraph"
    elif target == SpecialistRole.L2_AGENT:
        return "l2_subgraph"
    elif target == SpecialistRole.DBA_AGENT:
        return "dba_subgraph"
    return "end"
