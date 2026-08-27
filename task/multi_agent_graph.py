import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from checkpoint_factory import create_checkpointer

from bone import MultiAgentSupervisorState, SpecialistState, AgentStatus, SpecialistRole, AgentHandoff, parse_specialist_role
from supervisor import supervisor_node, route_after_supervisor, MAX_HANDOFF_COUNT, detect_deadlock
from specialist_agents import l2_subgraph, dba_subgraph
from l1_agent import l1_subgraph

logger = logging.getLogger("MultiAgentGraph")


async def p2p_handoff_router(state: MultiAgentSupervisorState, config: RunnableConfig) -> dict:
    """
    负责更新全局 Handoff 计数，检测死循环，并映射状态给目标 Specialist。
    """
    handoff = state.get("handoff_request")
    if not handoff:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "p2p_router", "error": "No handoff request"}]}

    # 1. 更新全局计数和历史
    new_count = state.get("handoff_count", 0) + 1
    history = state.get("handoff_history", []) + [handoff]

    update = {
        "handoff_count": new_count,
        "handoff_history": history,
        "current_specialist": handoff.to_agent,
        "_excluded_hypotheses": state.get("_excluded_hypotheses", []) + handoff.excluded_hypotheses,  # 累积已排除项
        "handoff_request": None  # 清空当前 request
    }

    # 2. 死循环/超限检测。如果触发，强制路由回 Supervisor 进行仲裁
    if new_count >= MAX_HANDOFF_COUNT or detect_deadlock(history):
        logger.warning(f"[P2P Router] Deadlock/Limit reached. Forcing Arbitration.")
        update["is_arbitrating"] = True
        return update  # 返回后，条件边将路由至 supervisor

    # 3. 正常 P2P：映射上下文给目标 Specialist
    update[
        "_task_for_specialist"] = f"[Handoff Context from {handoff.from_agent.value}]\n{handoff.context_summary}\nReason: {handoff.reason}"

    return update


def route_after_p2p_router(state: MultiAgentSupervisorState) -> str:
    """P2P 路由器的条件边"""
    if state.get("is_arbitrating"):
        return "supervisor"  # 触发仲裁，回到 Supervisor

    target = state.get("current_specialist")
    if target == SpecialistRole.L1_AGENT:
        return "l1_subgraph"
    elif target == SpecialistRole.L2_AGENT:
        return "l2_subgraph"
    elif target == SpecialistRole.DBA_AGENT:
        return "dba_subgraph"
    return "end"


async def l1_wrapper_node(state: MultiAgentSupervisorState, config: RunnableConfig) -> dict:
    """L1 子图包装节点（接入真实实现：ReAct 初步排查 + Handoff 评估）"""
    sub_state = {
        "task_description": state.get("_task_for_specialist", ""),
        "context_from_supervisor": state.get("topology_context", ""),
        "excluded_hypotheses": state.get("_excluded_hypotheses", []),
        "messages": []
    }
    l1_result = await l1_subgraph.ainvoke(sub_state, config)

    update = {
        "specialist_findings": {SpecialistRole.L1_AGENT.value: l1_result.get("specialist_findings", "")},
        "status": l1_result.get("status", AgentStatus.SUCCESS)
    }
    if l1_result.get("handoff_request"):
        update["handoff_request"] = l1_result["handoff_request"]
        update["status"] = AgentStatus.HANDOFF
    return update


async def l2_wrapper_node(state: MultiAgentSupervisorState, config: RunnableConfig) -> dict:
    """L2 子图包装节点"""
    sub_state = {
        "task_description": state.get("_task_for_specialist", ""),
        "context_from_supervisor": state.get("topology_context", ""),
        "excluded_hypotheses": state.get("_excluded_hypotheses", []),
        "messages": []
    }
    l2_result = await l2_subgraph.ainvoke(sub_state, config)

    update = {
        "specialist_findings": {SpecialistRole.L2_AGENT.value: l2_result.get("specialist_findings", "")},
        "status": l2_result.get("status", AgentStatus.SUCCESS)
    }
    if l2_result.get("handoff_request"):
        update["handoff_request"] = l2_result["handoff_request"]
        update["status"] = AgentStatus.HANDOFF
    return update


async def dba_wrapper_node(state: MultiAgentSupervisorState, config: RunnableConfig) -> dict:
    """DBA 子图包装节点"""
    sub_state = {
        "task_description": state.get("_task_for_specialist", ""),
        "context_from_supervisor": state.get("topology_context", ""),
        "excluded_hypotheses": state.get("_excluded_hypotheses", []),
        "messages": []
    }
    dba_result = await dba_subgraph.ainvoke(sub_state, config)

    update = {
        "specialist_findings": {SpecialistRole.DBA_AGENT.value: dba_result.get("specialist_findings", "")},
        "status": dba_result.get("status", AgentStatus.SUCCESS)
    }
    if dba_result.get("handoff_request"):
        update["handoff_request"] = dba_result["handoff_request"]
        update["status"] = AgentStatus.HANDOFF
    return update


def route_after_specialist(state: MultiAgentSupervisorState) -> str:
    """Specialist 执行后的路由：决定是 P2P 还是回 Supervisor"""
    status = state.get("status")
    handoff = state.get("handoff_request")

    # 如果触发了 Handoff，走 P2P 路由器
    if status == AgentStatus.HANDOFF and handoff:
        return "p2p_handoff_router"

    # 否则（SUCCESS/FAILED），回到 Supervisor 进行结果汇总
    return "supervisor"


async def build_multi_agent_graph():
    """构建多 Agent 协作主图 (支持 P2P 与仲裁)"""
    workflow = StateGraph(MultiAgentSupervisorState)

    # 1. 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("p2p_handoff_router", p2p_handoff_router)
    workflow.add_node("l1_subgraph", l1_wrapper_node)
    workflow.add_node("l2_subgraph", l2_wrapper_node)
    workflow.add_node("dba_subgraph", dba_wrapper_node)

    # 2. 设置入口
    workflow.set_entry_point("supervisor")

    # 3. Supervisor 路由 (初始分发 / 仲裁后分发)
    workflow.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {"l1_subgraph": "l1_subgraph", "l2_subgraph": "l2_subgraph", "dba_subgraph": "dba_subgraph", "end": END}
    )

    # 4. Specialist 路由 (P2P Handoff 或 返回 Supervisor)
    workflow.add_conditional_edges(
        "l1_subgraph", route_after_specialist,
        {"p2p_handoff_router": "p2p_handoff_router", "supervisor": "supervisor"}
    )
    workflow.add_conditional_edges(
        "l2_subgraph", route_after_specialist,
        {"p2p_handoff_router": "p2p_handoff_router", "supervisor": "supervisor"}
    )
    workflow.add_conditional_edges(
        "dba_subgraph", route_after_specialist,
        {"p2p_handoff_router": "p2p_handoff_router", "supervisor": "supervisor"}
    )

    # 5. P2P 路由器路由 (直接跳转目标 Specialist 或 强制仲裁)
    workflow.add_conditional_edges(
        "p2p_handoff_router", route_after_p2p_router,
        {
            "supervisor": "supervisor",
            "l1_subgraph": "l1_subgraph",
            "l2_subgraph": "l2_subgraph",
            "dba_subgraph": "dba_subgraph"
        }
    )

    checkpointer = await create_checkpointer()  # Redis 不可用时自动降级 MemorySaver
    return workflow.compile(checkpointer=checkpointer)


# 注意：构建已改为异步，需要时请 await build_multi_agent_graph()（不再提供模块级实例）
