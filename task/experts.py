import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from bone import ExpertState, AgentStatus
from tool import K8S_TOOLS, DB_TOOLS, TOOL_MAP

logger = logging.getLogger("ExpertAgents")

# ================= 配置常量 =================
MAX_TOOL_CALLS_PER_EXPERT = 5
MAX_LOG_LINES_KEEP = 10  # 日志保留行数
MAX_TOOL_OUTPUT_CHARS = 2000  # 超过此长度触发压缩

# ================= 专家 System Prompts =================
K8S_SYSTEM_PROMPT = """你是 Kubernetes 基础设施专家。
【重要约束】
1. 你只能访问 K8s 相关工具。
2. 请结合提供的【团队共享黑板】信息。
3. **防幻觉**：如实报告“未检测到异常”。
4. **防死循环**：如果连续两次调用相同工具且参数相同，请立即停止。
"""

DB_SYSTEM_PROMPT = """你是数据库性能专家。
【重要约束】
1. 你只能访问 DB 相关工具。
2. 请结合提供的【团队共享黑板】信息。
3. **防幻觉**：如实报告“未检测到异常”。
4. **防死循环**：如果连续两次调用相同工具且参数相同，请立即停止。
"""


def _compress_tool_output(tool_name: str, raw_output: str) -> str:
    """
    实时压缩工具输出
    1. 如果是日志，提取错误行和统计头。
    2. 如果是 JSON，保留关键字段。
    3. 截断过长文本。
    """
    if len(raw_output) <= MAX_TOOL_OUTPUT_CHARS:
        return raw_output

    logger.info(f"[Compressor] Compressing output for {tool_name} ({len(raw_output)} chars)")

    # 策略 1: 日志类压缩
    if "log" in tool_name.lower() or "trace" in raw_output.lower():
        lines = raw_output.split('\n')
        error_lines = [l for l in lines if any(kw in l for kw in ["ERROR", "Exception", "Fatal", "Crash"])]
        summary = f"[Log Summary: Total {len(lines)} lines, {len(error_lines)} errors]\n"
        if error_lines:
            summary += "\n".join(error_lines[:MAX_LOG_LINES_KEEP])
        else:
            summary += "\n".join(lines[-MAX_LOG_LINES_KEEP:])  # 保留最后几行
        return summary + "\n... [Output Truncated]"

    # 策略 2: JSON 类压缩
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict):
            # 保留前 3 个键值对作为预览
            preview = {k: v for i, (k, v) in enumerate(data.items()) if i < 3}
            return json.dumps(
                {"summary": "Large JSON object", "preview": preview, "note": "Full data omitted due to size"})
    except:
        pass

    # 策略 3: 通用截断
    return raw_output[:MAX_TOOL_OUTPUT_CHARS] + "\n... [Output Truncated]"


async def expert_executor_node(state: ExpertState, config: RunnableConfig, tools: List) -> dict:
    """
    通用专家执行节点逻辑 (ReAct Loop) - 优化版
    """
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"expert_findings": "Error: LLM not configured", "status": AgentStatus.FAILED}

    scratchpad_context = state.get("context_from_scratchpad", "")
    context_block = f"【重要背景信息 - 团队共享黑板】\n{scratchpad_context}" if scratchpad_context else "【无历史排查记录】"
    task_desc = state.get("task_description", "")

    system_content = f"{state.get('_system_prompt', '')}\n{context_block}\n【当前任务】\n{task_desc}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content="请开始排查。")
    ]

    llm_with_tools = llm.bind_tools(tools)
    tool_call_history: List[Tuple[str, str]] = []
    iteration_count = 0
    final_findings = ""

    try:
        while iteration_count < MAX_TOOL_CALLS_PER_EXPERT:
            iteration_count += 1
            logger.info(f"[Expert] ReAct Iteration {iteration_count}/{MAX_TOOL_CALLS_PER_EXPERT}")

            # 1. 调用 LLM
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # 2. 检查是否有工具调用
            if not response.tool_calls:
                final_findings = response.content
                break

            # 3. 执行工具调用
            new_messages = []
            for tc in response.tool_calls:
                tool_name = tc['name']
                tool_args = tc['args']
                tool_call_id = tc['id']

                # 死循环检测
                current_call_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
                if len(tool_call_history) >= 1 and tool_call_history[-1] == current_call_signature:
                    final_findings = f"检测到潜在死循环：连续两次调用相同工具 {tool_name}。排查终止。"
                    iteration_count = MAX_TOOL_CALLS_PER_EXPERT
                    break
                tool_call_history.append(current_call_signature)

                # 执行工具
                tool_fn = TOOL_MAP.get(tool_name)
                if tool_fn:
                    try:
                        result = await tool_fn.ainvoke(tool_args)
                        raw_content = str(result)

                        compressed_content = _compress_tool_output(tool_name, raw_content)

                        tool_msg = ToolMessage(content=compressed_content, tool_call_id=tool_call_id)
                        new_messages.append(tool_msg)
                    except Exception as e:
                        tool_msg = ToolMessage(content=f"执行失败: {str(e)[:500]}", tool_call_id=tool_call_id)
                        new_messages.append(tool_msg)
                else:
                    tool_msg = ToolMessage(content=f"错误: 未找到工具 {tool_name}", tool_call_id=tool_call_id)
                    new_messages.append(tool_msg)

            messages.extend(new_messages)

            if len(messages) > 10:
                # 保留第一条 (System) 和最后 9 条
                messages = [messages[0]] + messages[-9:]
                logger.debug("[Expert] Sliding window applied to messages.")

            if iteration_count >= MAX_TOOL_CALLS_PER_EXPERT and final_findings:
                break

        if not final_findings:
            summary_prompt = "基于以上的工具执行结果，请给出最终的排查总结。"
            summary_response = await llm.ainvoke(messages + [HumanMessage(content=summary_prompt)])
            final_findings = summary_response.content

    except Exception as e:
        logger.error(f"[Expert] Critical Error in ReAct Loop: {e}")
        final_findings = f"专家执行过程中发生严重错误: {str(e)}"

    return {
        "expert_findings": final_findings,
        "status": AgentStatus.SUCCESS
    }


def build_k8s_expert_graph():
    workflow = StateGraph(ExpertState)

    async def k8s_node(state: ExpertState, config: RunnableConfig):
        state['_system_prompt'] = K8S_SYSTEM_PROMPT
        return await expert_executor_node(state, config, K8S_TOOLS)

    workflow.add_node("execute", k8s_node)
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", END)
    return workflow.compile()


def build_db_expert_graph():
    workflow = StateGraph(ExpertState)

    async def db_node(state: ExpertState, config: RunnableConfig):
        state['_system_prompt'] = DB_SYSTEM_PROMPT
        return await expert_executor_node(state, config, DB_TOOLS)

    workflow.add_node("execute", db_node)
    workflow.set_entry_point("execute")
    workflow.add_edge("execute", END)
    return workflow.compile()


k8s_expert_subgraph = build_k8s_expert_graph()
db_expert_subgraph = build_db_expert_graph()