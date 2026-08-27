import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from bone import SpecialistState, AgentStatus, SpecialistRole, AgentHandoff, parse_specialist_role
from tool import K8S_TOOLS, TOOL_MAP
from utils import to_structured_output_schema

logger = logging.getLogger("L1Agent")

L1_SYSTEM_PROMPT = """你是 L1 初级运维专家 (First Responder)。
【职责】
1. 快速响应告警，进行初步信息收集
2. 查看应用日志，识别明显的错误模式
3. 查询基础监控指标（CPU、内存、网络）
4. 判断问题是否属于已知模式，可以立即解决

【Handoff 触发条件】
- 发现复杂的代码逻辑问题 → Handoff 给 L2_AGENT
- 发现数据库相关问题（慢查询、连接池）→ Handoff 给 DBA_AGENT
- 问题超出你的能力范围或需要更深层分析

【工具使用】
你可以使用所有 K8s 相关工具进行日志查询和 Pod 状态检查。
"""

MAX_TOOL_CALLS = 8


async def l1_executor_node(state: SpecialistState, config: RunnableConfig) -> dict:
    """L1 Agent 执行节点"""
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {
            "specialist_findings": "Error: LLM not configured",
            "status": AgentStatus.FAILED
        }
    
    context = state.get("context_from_supervisor", "")
    task_desc = state.get("task_description", "")
    excluded = state.get("excluded_hypotheses", [])
    
    # 构建系统提示
    excluded_str = "\n".join([f"- {h}" for h in excluded]) if excluded else "无"
    system_content = f"{L1_SYSTEM_PROMPT}\n\n【上下文与任务】\n{context}\n\n【已排除的假设】\n{excluded_str}\n\n{task_desc}"
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content="请开始初步排查和信息收集。")
    ]
    
    llm_with_tools = llm.bind_tools(K8S_TOOLS)
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
            
            # 执行工具调用
            new_messages = []
            for tc in response.tool_calls:
                tool_fn = TOOL_MAP.get(tc['name'])
                if tool_fn:
                    try:
                        result = await tool_fn.ainvoke(tc['args'])
                        # 压缩输出
                        result_str = str(result)
                        if len(result_str) > 3000:
                            result_str = result_str[:3000] + "\n... [Truncated]"
                        new_messages.append(
                            ToolMessage(content=result_str, tool_call_id=tc['id'])
                        )
                    except Exception as e:
                        new_messages.append(
                            ToolMessage(
                                content=f"工具执行失败: {str(e)[:500]}",
                                tool_call_id=tc['id']
                            )
                        )
                else:
                    new_messages.append(
                        ToolMessage(
                            content=f"未找到工具 {tc['name']}",
                            tool_call_id=tc['id']
                        )
                    )
            messages.extend(new_messages)
        
        if not final_findings:
            summary_response = await llm.ainvoke(
                messages + [HumanMessage(content="请给出初步排查总结。")]
            )
            final_findings = summary_response.content
    
    except Exception as e:
        logger.error(f"[L1] Execution error: {e}")
        final_findings = f"L1 执行过程中发生错误: {str(e)}"
    
    return {
        "specialist_findings": final_findings,
        "messages": messages,
        "status": AgentStatus.SUCCESS
    }


async def l1_handoff_evaluation(state: SpecialistState, config: RunnableConfig) -> dict:
    """L1 Handoff 评估"""
    llm = config.get("configurable", {}).get("llm_instance")
    findings = state.get("specialist_findings", "")
    
    if not findings or not llm:
        return {"status": AgentStatus.SUCCESS}
    
    eval_prompt = """作为 L1 初级运维，你刚完成初步排查。
【排查发现】
{findings}

【判断任务】
1. 问题是否已明确定位且在你的能力范围内？如果是，不需要 Handoff。
2. 如果发现是复杂的代码逻辑问题，Handoff 给 L2_AGENT。
3. 如果发现是数据库问题（慢查询、连接池），Handoff 给 DBA_AGENT。
4. 总结你已排除的假设。

输出 JSON:
{{
    "need_handoff": true或false,
    "target_agent": "L2_AGENT"或"DBA_AGENT"或null,
    "context_summary": "如果需要 handoff，总结发现的线索",
    "reason": "发起 handoff 的原因",
    "excluded_hypotheses": ["已排除的假设1", "已排除的假设2"]
}}
"""
    
    try:
        structured_llm = llm.with_structured_output(to_structured_output_schema({
            "need_handoff": bool,
            "target_agent": str,
            "context_summary": str,
            "reason": str,
            "excluded_hypotheses": List[str]
        }), method="json_mode")
        
        response = await structured_llm.ainvoke([
            SystemMessage(content=eval_prompt.format(findings=findings)),
            HumanMessage(content="请评估是否需要 Handoff。")
        ])
        
        # 健壮解析目标角色（LLM 输出大写枚举名，解析失败不触发交接）
        target_role = parse_specialist_role(response.get("target_agent"))
        if response.get("need_handoff") and target_role:
            logger.info(f"[L1] Triggering Handoff to {target_role.value}")
            handoff = AgentHandoff(
                from_agent=SpecialistRole.L1_AGENT,
                to_agent=target_role,
                context_summary=response.get("context_summary", ""),
                reason=response.get("reason", ""),
                excluded_hypotheses=response.get("excluded_hypotheses", [])
            )
            return {"handoff_request": handoff, "status": AgentStatus.HANDOFF}
    
    except Exception as e:
        logger.warning(f"[L1] Handoff evaluation failed: {e}")
    
    return {"status": AgentStatus.SUCCESS}


def build_l1_subgraph():
    """构建 L1 Agent 子图"""
    workflow = StateGraph(SpecialistState)
    
    async def l1_execute(state: SpecialistState, config: RunnableConfig):
        return await l1_executor_node(state, config)
    
    async def l1_evaluate(state: SpecialistState, config: RunnableConfig):
        if state.get("status") == AgentStatus.SUCCESS:
            return await l1_handoff_evaluation(state, config)
        return {}
    
    workflow.add_node("execute", l1_execute)
    workflow.add_node("evaluate_handoff", l1_evaluate)
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", "evaluate_handoff")
    workflow.add_edge("evaluate_handoff", END)
    
    return workflow.compile()


l1_subgraph = build_l1_subgraph()
