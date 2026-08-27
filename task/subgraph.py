"""
工具执行子图 (Tool Execution Subgraph)
完善了对 interrupt 返回值的处理，支持外部注入的拒绝信号。

【关键修复】子图状态与主图对齐：
- 工具执行结果写入 messages 字段（原实现写 tool_messages，主图状态未声明该字段，
  结果被静默丢弃，导致 executor 永远收不到工具返回、反复重发同一步调用）。
- messages 使用 add_messages reducer：子图内的 ToolMessage 与主图消息流按 id 去重合并，
  executor 下一轮即可在 messages 末尾看到 ToolMessage 进入评估相。
- ToolExecutionState 改为标准 TypedDict（原 Dict 子类注解不生效）。
"""
import asyncio
import logging
from typing import List, Dict, Annotated, TypedDict

from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from tool import TOOL_MAP, DANGEROUS_TOOLS, TOOL_TIMEOUTS
from utils import get_circuit_breaker

logger = logging.getLogger("ToolSubgraph")


class ToolExecutionState(TypedDict, total=False):
    pending_tool_calls: List[Dict]
    messages: Annotated[list[BaseMessage], add_messages]  # 与主图 messages 同名，结果可合并回主图
    approved_tools: List[str]


async def hitl_check_node(state: ToolExecutionState) -> dict:
    """
    人工审批节点
    """
    pending = state.get("pending_tool_calls", [])
    approved_list = state.get("approved_tools", [])
    messages_to_return = []
    remaining_pending = []

    for tc in pending:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        # 1. Plan 级批量授权检查
        if tool_name in approved_list:
            remaining_pending.append(tc)
            continue

        # 2. 危险工具拦截
        if tool_name in DANGEROUS_TOOLS:
            logger.info(f"[HITL] 检测到高危工具: {tool_name}，请求人工审批...")

            approval = interrupt({
                "action": "approval_required",
                "tool_name": tool_name,
                "args": tool_args,
                "message": f"请求执行高危操作: {tool_name}。请确认是否继续？"
            })

            if isinstance(approval, dict) and approval.get("approved"):
                logger.info(f"[HITL] 工具 {tool_name} 已批准。")
                remaining_pending.append(tc)
            else:
                reason = "未知原因"
                if isinstance(approval, dict):
                    reason = approval.get("reason", "用户拒绝")
                else:
                    reason = "无效响应"

                logger.warning(f"[HITL] 工具 {tool_name} 被拒绝: {reason}")
                messages_to_return.append(ToolMessage(
                    content=f"操作已被拒绝: {reason}",
                    tool_call_id=tool_call_id
                ))
        else:
            remaining_pending.append(tc)

    return {
        "pending_tool_calls": remaining_pending,
        "messages": messages_to_return
    }


async def circuit_breaker_node(state: ToolExecutionState) -> dict:
    pending = state.get("pending_tool_calls", [])
    new_msgs = []
    valid_pending = []

    for tc in pending:
        tool_name = tc["name"]
        cb = get_circuit_breaker(tool_name)
        if cb.is_open():
            new_msgs.append(ToolMessage(
                content=f"[CIRCUIT_BREAKER] 工具 {tool_name} 已熔断，跳过执行。",
                tool_call_id=tc["id"]
            ))
        else:
            valid_pending.append(tc)

    return {
        "pending_tool_calls": valid_pending,
        "messages": new_msgs
    }


async def actual_tool_execution_node(state: ToolExecutionState) -> dict:
    pending = state.get("pending_tool_calls", [])
    new_msgs = []

    for tc in pending:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]
        timeout_seconds = TOOL_TIMEOUTS.get(tool_name, 10)
        cb = get_circuit_breaker(tool_name)
        tool_fn = TOOL_MAP.get(tool_name)

        if not tool_fn:
            new_msgs.append(ToolMessage(content=f"错误: 未找到工具 {tool_name}", tool_call_id=tool_call_id))
            continue

        try:
            result = await asyncio.wait_for(tool_fn.ainvoke(tool_args), timeout=timeout_seconds)
            cb.record_success()
            new_msgs.append(ToolMessage(content=str(result)[:1000], tool_call_id=tool_call_id))
        except asyncio.TimeoutError:
            cb.record_failure()
            new_msgs.append(ToolMessage(content=f"执行超时: {tool_name}", tool_call_id=tool_call_id))
        except Exception as e:
            cb.record_failure()
            new_msgs.append(ToolMessage(content=f"执行失败: {str(e)[:500]}", tool_call_id=tool_call_id))

    return {
        "pending_tool_calls": [],
        "messages": new_msgs
    }


def build_tool_execution_subgraph():
    subgraph = StateGraph(ToolExecutionState)
    subgraph.add_node("hitl_check", hitl_check_node)
    subgraph.add_node("circuit_breaker", circuit_breaker_node)
    subgraph.add_node("execute", actual_tool_execution_node)

    subgraph.set_entry_point("hitl_check")
    subgraph.add_edge("hitl_check", "circuit_breaker")
    subgraph.add_edge("circuit_breaker", "execute")
    subgraph.add_edge("execute", END)

    return subgraph.compile()
