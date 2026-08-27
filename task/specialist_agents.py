import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from bone import SpecialistState, AgentStatus, SpecialistRole, AgentHandoff, parse_specialist_role
from tool import K8S_TOOLS, DB_TOOLS, TOOL_MAP
from utils import to_structured_output_schema

logger = logging.getLogger("SpecialistAgents")

MAX_TOOL_CALLS = 5
MAX_TOOL_OUTPUT_CHARS = 2000

#在 System Prompt 中强调已排除项
L2_SYSTEM_PROMPT = """你是 Kubernetes 与微服务架构资深专家 (L2_Agent)。
【职责】负责深度代码分析、链路追踪。只能使用 K8s 相关工具。
【⚠️ 关键约束】：你必须严格遵守【已排除的假设】列表，绝对不允许重复排查已排除的问题！
【Handoff 触发条件】：如果你明确发现根因在于数据库层（如：SQL 慢查询、死锁、连接池耗尽），你必须停止排查，准备发起 Handoff 给 DBA_AGENT，并总结你已排除的假设。
"""

DBA_SYSTEM_PROMPT = """你是数据库性能与架构专家 (DBA_Agent)。
【职责】专门负责 SQL 慢查询、锁竞争、连接池诊断。只能使用 DB 相关工具。
【⚠️ 关键约束】：你必须严格遵守【已排除的假设】列表，绝对不允许重复排查已排除的问题！结合 Handoff 上下文进行深度诊断。
"""

#Handoff 评估 Prompt，强制要求输出 excluded_hypotheses
HANDOFF_EVALUATION_PROMPT = """作为 {agent_role}，你刚刚完成了一轮排查。
【排查发现】
{findings}

【判断任务】
1. 根因是否明确属于其他专家的专业领域？如果是，准备 Handoff。
2. 【关键】总结你在排查过程中已经验证过并排除的假设（例如："已排除网络层丢包问题"、"已排除应用层 OOM"），填入 excluded_hypotheses。

请以 JSON 格式输出：
{{
    "need_handoff": true | false,
    "target_agent": "DBA_AGENT" | "L2_AGENT" | null,
    "context_summary": "如果需要 handoff，总结你发现的线索和当前卡点",
    "reason": "发起 handoff 的具体原因",
    "excluded_hypotheses": ["已排除的假设1", "已排除的假设2"]
}}
"""


def _compress_tool_output(tool_name: str, raw_output: str) -> str:
    if len(raw_output) <= MAX_TOOL_OUTPUT_CHARS: return raw_output
    if "log" in tool_name.lower():
        lines = raw_output.split('\n')
        error_lines = [l for l in lines if any(kw in l for kw in ["ERROR", "Exception", "Fatal"])]
        return f"[Log Summary]\n" + "\n".join(error_lines[:10]) + "\n... [Truncated]"
    return raw_output[:MAX_TOOL_OUTPUT_CHARS] + "\n... [Truncated]"


async def specialist_executor_node(state: SpecialistState, config: RunnableConfig, tools: List,
                                   system_prompt: str) -> dict:
    """通用 Specialist ReAct 执行节点"""
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm: return {"specialist_findings": "Error: LLM not configured", "status": AgentStatus.FAILED}

    context = state.get("context_from_supervisor", "")
    task_desc = state.get("task_description", "")

    #注入已排除项到 Prompt
    excluded = state.get("excluded_hypotheses", [])
    excluded_str = "\n".join([f"- {h}" for h in excluded]) if excluded else "无"

    system_content = f"{system_prompt}\n\n【上下文与任务】\n{context}\n\n【已排除的假设 (严禁重复排查)】\n{excluded_str}\n\n{task_desc}"

    messages = [SystemMessage(content=system_content), HumanMessage(content="请开始排查。")]
    llm_with_tools = llm.bind_tools(tools)
    iteration_count = 0
    final_findings = ""

    try:
        while iteration_count < MAX_TOOL_CALLS:
            iteration_count += 1
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                final_findings = response.content
                break

            new_messages = []
            for tc in response.tool_calls:
                tool_fn = TOOL_MAP.get(tc['name'])
                if tool_fn:
                    try:
                        result = await tool_fn.ainvoke(tc['args'])
                        new_messages.append(
                            ToolMessage(content=_compress_tool_output(tc['name'], str(result)), tool_call_id=tc['id']))
                    except Exception as e:
                        new_messages.append(ToolMessage(content=f"执行失败: {str(e)[:500]}", tool_call_id=tc['id']))
                else:
                    new_messages.append(ToolMessage(content=f"未找到工具 {tc['name']}", tool_call_id=tc['id']))
            messages.extend(new_messages)

        if not final_findings:
            summary_response = await llm.ainvoke(messages + [HumanMessage(content="请给出最终的排查总结。")])
            final_findings = summary_response.content

    except Exception as e:
        final_findings = f"执行过程中发生严重错误: {str(e)}"

    return {"specialist_findings": final_findings, "messages": messages, "status": AgentStatus.SUCCESS}


async def handoff_evaluation_node(state: SpecialistState, config: RunnableConfig, agent_role: SpecialistRole) -> dict:
    """评估是否需要触发 Handoff """
    llm = config.get("configurable", {}).get("llm_instance")
    findings = state.get("specialist_findings", "")

    if not findings or not llm: return {"status": AgentStatus.SUCCESS}

    try:
        structured_llm = llm.with_structured_output(to_structured_output_schema({
            "need_handoff": bool, "target_agent": Optional[str],
            "context_summary": str, "reason": str, "excluded_hypotheses": List[str]
        }), method="json_mode")

        response = await structured_llm.ainvoke([
            SystemMessage(content=HANDOFF_EVALUATION_PROMPT.format(agent_role=agent_role.value, findings=findings)),
            HumanMessage(content="请评估是否需要 Handoff。")
        ])

        # 健壮解析目标角色（LLM 输出大写枚举名，解析失败不触发交接）
        target_role = parse_specialist_role(response.get("target_agent"))
        if response.get("need_handoff") and target_role:
            logger.info(f"[{agent_role.value}] Triggering Handoff to {target_role.value}")
            handoff = AgentHandoff(
                from_agent=agent_role,
                to_agent=target_role,
                context_summary=response.get("context_summary", ""),
                reason=response.get("reason", ""),
                excluded_hypotheses=response.get("excluded_hypotheses", [])
            )
            return {"handoff_request": handoff, "status": AgentStatus.HANDOFF}

    except Exception as e:
        logger.warning(f"[{agent_role.value}] Handoff evaluation failed: {e}")

    return {"status": AgentStatus.SUCCESS}


def build_l2_subgraph():
    workflow = StateGraph(SpecialistState)

    async def l2_execute(state: SpecialistState, config: RunnableConfig):
        return await specialist_executor_node(state, config, K8S_TOOLS, L2_SYSTEM_PROMPT)

    async def l2_evaluate(state: SpecialistState, config: RunnableConfig):
        if state.get("status") == AgentStatus.SUCCESS:
            return await handoff_evaluation_node(state, config, SpecialistRole.L2_AGENT)
        return {}

    workflow.add_node("execute", l2_execute)
    workflow.add_node("evaluate_handoff", l2_evaluate)
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", "evaluate_handoff")
    workflow.add_edge("evaluate_handoff", END)
    return workflow.compile()


def build_dba_subgraph():
    workflow = StateGraph(SpecialistState)

    async def dba_execute(state: SpecialistState, config: RunnableConfig):
        return await specialist_executor_node(state, config, DB_TOOLS, DBA_SYSTEM_PROMPT)

    workflow.add_node("execute", dba_execute)
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", END)
    return workflow.compile()


l2_subgraph = build_l2_subgraph()
dba_subgraph = build_dba_subgraph()
