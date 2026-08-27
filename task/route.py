import json
import logging
import asyncio
import re
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.exceptions import OutputParserException

from bone import AgentStatus, OpsAgentState
from retrieve import rag_retrieval_node
from tool import ALL_TOOLS, DANGEROUS_TOOLS, TOOL_MAP
from nodes import (router_node, generator_node, chitchat_node, clarify_node, fallback_node)
from rca_generator import rca_generator_node
from rca_renderer import distribute_and_ingest_rca
from graph_expansion import graph_expansion_node
from topology_node import graph_query_node
from critic import critic_node
from utils import retry_with_backoff, to_structured_output_schema, pydantic_model_to_json_schema
from subgraph import build_tool_execution_subgraph
from reflection_node import reflect_and_adjust_node
from audit import audit_node
from settings import get_redis_config
from checkpoint_factory import create_checkpointer

logger = logging.getLogger("route")


def load_redis_config_from_ini(ini_path: str = "config.ini") -> dict:
    """兼容保留：实际读取已收敛到 settings.get_redis_config（密码只来自 .env）"""
    return get_redis_config(ini_path)


class PlanStep(BaseModel):
    step_id: int = Field(description="步骤序号")
    action: str = Field(description="要执行的动作")
    target: str = Field(description="操作目标")
    purpose: str = Field(description="这一步的目的")


class TroubleshootingPlan(BaseModel):
    root_cause_hypothesis: str = Field(description="初步根因假设")
    steps: List[PlanStep] = Field(description="排查步骤列表")


PLANNER_SYSTEM_PROMPT = """你是一个高级 SRE 专家。你的任务是根据用户的故障描述，生成一个结构化的排查计划 (SOP)，请以 JSON 格式输出。
【关键约束】每个排查步骤的 action 必须严格使用【可用工具清单】中的工具名（如 get_pod_status、query_slow_sql），禁止使用自然语言命令；target 必须是合法资源标识符（英文服务名、Pod 名等）。
【可用工具清单】
{tool_inventory}
注意：如果计划中包含高危操作（如重启服务、扩容），请在计划中标记出来。
"""


def _normalize_plan_dict(raw) -> dict:
    """字段别名归一化：json_mode 下模型不强制遵守 schema 字段名，
    把常见别名映射到 TroubleshootingPlan 规范字段，缺失字段给安全默认值。"""
    if not isinstance(raw, dict):
        raw = {}
    steps_raw = raw.get("steps") or []
    if not isinstance(steps_raw, list):
        steps_raw = []
    norm_steps = []
    for idx, step in enumerate(steps_raw):
        if not isinstance(step, dict):
            step = {}
        step_id = step.get("step_id", step.get("step", step.get("id", idx + 1)))
        if not isinstance(step_id, int):
            try:
                step_id = int(step_id)
            except (TypeError, ValueError):
                step_id = idx + 1
        action = step.get("action") or step.get("tool") or step.get("name") or ""
        target = step.get("target") or step.get("service") or step.get("pod") or step.get("object") or ""
        purpose = step.get("purpose") or step.get("goal") or step.get("description") or str(action)
        norm_steps.append({
            "step_id": step_id,
            "action": str(action),
            "target": str(target),
            "purpose": str(purpose),
        })
    return {
        "root_cause_hypothesis": str(raw.get("root_cause_hypothesis") or raw.get("hypothesis") or "Unknown"),
        "steps": norm_steps,
    }


async def _parse_plan_with_fallback(llm, messages):
    structured_llm = llm.with_structured_output(pydantic_model_to_json_schema(TroubleshootingPlan), method="json_mode")
    try:
        raw = await structured_llm.ainvoke(messages); return TroubleshootingPlan(**_normalize_plan_dict(raw)), None
    except OutputParserException as e:
        try:
            fix_prompt_messages = messages + [HumanMessage(content=f"解析失败: {e}\n请重新输出标准 JSON。")]
            raw = await structured_llm.ainvoke(fix_prompt_messages); return TroubleshootingPlan(**_normalize_plan_dict(raw)), None
        except:
            try:
                response = await llm.ainvoke(messages + [HumanMessage(content="请直接输出 JSON 格式的 Plan。")])
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    plan_dict = _normalize_plan_dict(json.loads(json_match.group()))
                    steps = [PlanStep(**s) for s in plan_dict.get('steps', [])]
                    return TroubleshootingPlan(root_cause_hypothesis=plan_dict.get('root_cause_hypothesis', 'Unknown'),
                                               steps=steps), None
            except:
                pass
    return None, "STRUCTURED_PARSE_FAILED"


@retry_with_backoff(max_retries=2, base_delay=1.0, exceptions=(Exception,))
async def planner_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm: return {"status": AgentStatus.FAILED}

    query = state.get("query", "")
    execution_history = state.get("execution_history", [])
    next_strategy_hint = state.get("next_strategy_hint", "")

    # === 读取失败路径黑名单 ===
    failed_paths = state.get("failed_paths", [])
    blacklist_prompt = ""
    if failed_paths:
        unique_failed = list(set(failed_paths[-5:]))  # 取最近5个去重
        blacklist_prompt = f"\n\n【历史失败路径黑名单 (严禁再次规划以下路径)】:\n" + "\n".join(
            [f"- {p}" for p in unique_failed])
        logger.warning(f"[Planner] Injecting blacklist into prompt: {unique_failed}")

    history_str = ""
    if execution_history:
        history_items = [
            f"Step {i + 1}: {h['step_info']['action']} on {h['step_info']['target']} ({h['result_summary'][:50]})" for
            i, h in enumerate(execution_history)]
        history_str = "\n【已执行步骤】:\n" + "\n".join(history_items)

    hint_str = f"\n\n【系统强制修正策略】: {next_strategy_hint}\n请基于此策略重新规划排查路径。" if next_strategy_hint else ""

    # 拼接黑名单约束
    human_content = f"故障描述: {query}\n{history_str}{hint_str}{blacklist_prompt}\n请以 JSON 格式生成排查计划，必须包含字段：root_cause_hypothesis（初步根因假设，字符串）和 steps（数组，每个元素包含 step_id 整数、action 工具名、target 操作目标、purpose 该步目的）。"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(tool_inventory="\n".join(
            f"- {t.name}: {(t.description or '').strip().splitlines()[0] if getattr(t, 'description', None) else ''}"
            for t in ALL_TOOLS) or "（无可用工具）")),
        HumanMessage(content=human_content)
    ]

    try:
        plan, error = await _parse_plan_with_fallback(llm, messages)
        if error: return {"status": AgentStatus.FAILED, "error_log": [{"node": "planner", "error": "Parse Failed"}]}

        dangerous_steps_in_plan = [s for s in plan.steps if s.action in DANGEROUS_TOOLS]
        approved_tools_list = []

        if dangerous_steps_in_plan:
            tool_names = list(set([s.action for s in dangerous_steps_in_plan]))
            approval_response = interrupt({
                "action": "plan_approval_required", "plan_summary": plan.root_cause_hypothesis,
                "dangerous_tools": tool_names, "message": f"计划中包含高危操作: {tool_names}。是否授权？"
            })
            if approval_response.get("approved"):
                approved_tools_list = tool_names
            else:
                return {"status": AgentStatus.FAILED, "final_answer": "计划被拒绝。", "system_hints": ["⚠️ 计划被拒绝"]}

        plan_dict = plan.model_dump()
        return {
            "current_plan": plan_dict["steps"], "current_step": 0, "approved_tools": approved_tools_list,
            "system_hints": [f"初始假设: {plan_dict['root_cause_hypothesis']}"],
            "status": AgentStatus.RUNNING, "messages": [AIMessage(content=f"已生成排查计划。")],
            "next_strategy_hint": ""  # 清空 hint
        }
    except Exception as e:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "planner", "error": str(e)}]}


EXECUTOR_EVAL_PROMPT = """你是一个 SRE 评估专家。请根据当前步骤的目的和工具执行结果，评估下一步行动。

当前步骤目的: {purpose}
工具执行结果: 
{result}

请分析结果是否达到了当前步骤的目的，并输出 JSON 格式：
{{
    "decision": "NEXT_STEP" | "REPLAN" | "FINISH",
    "reason": "简要说明理由。如果 FINISH，说明根因已找到；如果 REPLAN，说明当前路径错误需要重新规划。"
}}
"""


async def executor_node(state: Dict[str, Any], config: RunnableConfig) -> dict:
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": "FAILED", "error": "LLM not configured"}

    current_plan = state.get("current_plan", [])
    step_index = state.get("current_step", 0)
    messages = state.get("messages", [])
    execution_history = state.get("execution_history", [])

    if current_plan and step_index >= len(current_plan):
        logger.info("[Executor] All steps completed. Proceeding to Critic.")
        return {"status": "SUCCESS", "next_action": "critic_review"}

    if not current_plan:
        replan_count = state.get("replan_count", 0) + 1; return {"status": AgentStatus.FAILED, "replan_count": replan_count, "error_log": [{"node": "executor", "error": "Replan limit reached"}]} if replan_count >= 3 else {"status": "RUNNING", "next_action": "replan", "replan_count": replan_count}

    current_step = current_plan[step_index]
    action = current_step.get("action")
    target = current_step.get("target")
    purpose = current_step.get("purpose")

    if messages and isinstance(messages[-1], ToolMessage):
        tool_result = messages[-1].content
        logger.info(f"[Executor] Evaluating tool result for step {step_index + 1}...")

        try:
            structured_llm = llm.with_structured_output(to_structured_output_schema({
                "decision": str,
                "reason": str
            }), method="json_mode")
            eval_response = await structured_llm.ainvoke([
                SystemMessage(content="你是一个严谨的 SRE 评估专家。"),
                HumanMessage(content=EXECUTOR_EVAL_PROMPT.format(purpose=purpose, result=tool_result[:2000]))
            ])

            decision = eval_response.get("decision", "NEXT_STEP")
            reason = eval_response.get("reason", "")

            completed_step_record = {
                "step_info": current_step,
                "result_summary": tool_result[:500],
                "status": "completed",
                "eval_reason": reason
            }
            updated_history = execution_history + [completed_step_record]

            if decision == "FINISH":
                logger.info(f"[Executor] Root cause found early. Reason: {reason}")
                return {
                    "execution_history": updated_history,
                    "status": "SUCCESS",
                    "next_action": "critic_review",
                    "system_hints": [f"提前发现根因: {reason}"]
                }
            elif decision == "REPLAN":
                logger.warning(f"[Executor] Triggering replan. Reason: {reason}")
                # 把当前失败路径写入黑名单（与反思节点格式一致），防止重规划走同一条死路
                failed_action = f"{current_step.get('action')}:{current_step.get('target')}"
                return {
                    "execution_history": updated_history,
                    "current_plan": [],
                    "current_step": 0,
                    "status": "RUNNING",
                    "next_action": "replan",
                    "failed_paths": state.get("failed_paths", []) + [failed_action],
                    "replan_count": state.get("replan_count", 0) + 1,
                    "system_hints": [f"触发重规划: {reason}"]
                }
            else:
                logger.info(f"[Executor] Step {step_index + 1} evaluated as success. Moving to next step.")
                return {
                    "execution_history": updated_history,
                    "current_step": step_index + 1,
                    "status": "RUNNING",
                    "next_action": "execute_next"
                }

        except Exception as e:
            logger.error(f"[Executor] Evaluation failed: {e}. Defaulting to NEXT_STEP.")
            return {
                "current_step": step_index + 1,
                "status": "RUNNING",
                "next_action": "execute_next"
            }

    logger.info(f"[Executor] Generating tool call for Step {step_index + 1}: {action} on {target}")

    prompt = f"当前步骤: {action} on {target}\n目的: {purpose}\n请使用可用的工具完成此步骤。"
    context_messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state.get("query", ""))
    ]

    llm_with_tools = llm.bind_tools(state.get("available_tools", []))
    response: AIMessage = await llm_with_tools.ainvoke(context_messages)

    if response.tool_calls:
        return {
            "messages": [response],
            "pending_tool_calls": response.tool_calls,
            "next_action": "execute_next",
            "status": "RUNNING"
        }
    else:
        new_history_entry = {
            "step_info": current_step,
            "result_summary": response.content[:200] if response.content else "No output",
            "status": "completed"
        }
        return {
            "messages": [response],
            "execution_history": execution_history + [new_history_entry],
            "current_step": step_index + 1,
            "status": "RUNNING"
        }


async def build_graph(redis_config: Optional[dict] = None, ini_path: str = "config.ini"):
    graph = StateGraph(OpsAgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("audit", audit_node)
    graph.add_node("graph_expansion", graph_expansion_node)
    graph.add_node("topology_query", graph_query_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("rca_generator", rca_generator_node)
    graph.add_node("distributor", distribute_and_ingest_rca)
    graph.add_node("generator_node", generator_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("tool_executor", build_tool_execution_subgraph())
    graph.add_node("reflect_and_adjust", reflect_and_adjust_node)  # 注册反思节点

    graph.set_entry_point("router")

    def route_after_router(state: OpsAgentState) -> str:
        action = state.get("next_action", "RAG_SEARCH")
        return {"RAG_SEARCH": "graph_expansion", "TOOL_EXECUTION": "topology_query", "CHITCHAT": "chitchat",
                "CLARIFY": "clarify"}.get(action, "graph_expansion")

    graph.add_conditional_edges("router", route_after_router,
                                {"graph_expansion": "graph_expansion", "topology_query": "topology_query",
                                 "chitchat": "chitchat", "clarify": "clarify"})
    graph.add_edge("graph_expansion", "rag_retrieval")

    def route_after_topology(state: OpsAgentState) -> str:
        return "planner" if state.get("next_action") == "TOOL_EXECUTION" else "graph_expansion"

    graph.add_conditional_edges("topology_query", route_after_topology,
                                {"planner": "planner", "graph_expansion": "graph_expansion"})

    def route_after_rag(state: OpsAgentState) -> str:
        return "planner" if state.get("next_action") == "TOOL_EXECUTION" else "generator_node"

    graph.add_edge("rag_retrieval", "audit")
    graph.add_conditional_edges("audit", route_after_rag,
                                {"planner": "planner", "generator_node": "generator_node"})

    graph.add_edge("planner", "executor")

    def route_after_executor(state: OpsAgentState) -> str:
        if state.get("status") == AgentStatus.FAILED: return "fallback"
        if state.get("next_action") == "critic_review": return "critic"
        # REPLAN：计划已被清空，回到 planner 重新规划（带失败路径黑名单）
        if state.get("next_action") == "replan": return "fallback" if state.get("replan_count", 0) >= 3 else "planner"  # 重规划超 3 次转兜底
        if state.get("pending_tool_calls"): return "tool_executor"
        return "executor"

    graph.add_conditional_edges("executor", route_after_executor,
                                {"tool_executor": "tool_executor", "critic": "critic", "planner": "planner", "executor": "executor",
                                 "fallback": "fallback"})

    # === 修改：Tool Executor 进入 Reflection ===
    graph.add_edge("tool_executor", "reflect_and_adjust")

    # === 新增：Reflection 节点的路由逻辑 ===
    def route_after_reflection(state: OpsAgentState) -> str:
        if state.get("status") == AgentStatus.FAILED: return "fallback"
        # 如果触发了策略重置，current_plan 会被清空，且存在 next_strategy_hint
        if not state.get("current_plan") and state.get("next_strategy_hint"):
            return "planner"
        return "executor"

    graph.add_conditional_edges("reflect_and_adjust", route_after_reflection,
                                {"planner": "planner", "executor": "executor", "fallback": "fallback"})

    graph.add_edge("critic", "rca_generator")
    graph.add_edge("rca_generator", "distributor")
    graph.add_edge("distributor", END)
    graph.add_edge("generator_node", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("clarify", END)
    graph.add_edge("fallback", END)

    checkpointer = await create_checkpointer(ini_path)  # Redis 不可用时自动降级 MemorySaver
    return graph.compile(checkpointer=checkpointer)
