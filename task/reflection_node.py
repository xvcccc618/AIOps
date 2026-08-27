# reflection_node.py
import logging
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from bone import OpsAgentState, AgentStatus
from memory_manager import get_memory_manager
from reflection_schema import ReflectionResult
from utils import pydantic_model_to_json_schema

logger = logging.getLogger("ReflectionNode")

REFLECTION_PROMPT = """你是一个高级 SRE 故障排查专家。请根据当前的排查历史、工具执行结果以及系统检测到的异常，进行深度反思。

【当前排查历史摘要】
{stm_summary}

【系统死胡同检测结果】
{dead_end_detection}

【当前假设与证据】
当前假设: {current_hypothesis}
收集到的证据: {evidence_summary}

【任务】
1. 评估当前假设的证据支持度 (0.0 - 1.0)。
2. 判断是否陷入死胡同。
3. 对刚才的排查步骤进行批评。
4. 制定下一步的修正策略。
5. 【关键】列出支撑你反思结论的客观证据摘要（如：'check_metrics 返回 CPU 99%'），严禁主观臆断。

请以 JSON 格式输出：
{{
    "current_hypothesis": "更新或维持当前的假设",
    "evidence_support_score": 0.0到1.0之间的浮点数,
    "is_stuck": true或false,
    "critique": "对刚才操作的批评",
    "next_strategy": "下一步的具体修正策略",
    "evidence_citations": ["证据1摘要", "证据2摘要"]
}}
"""


def _detect_dead_end(stm_events: List[Any]) -> Dict[str, Any]:
    """任务 3：死胡同检测算法"""
    result = {"repeated_action_detected": False, "invalid_feedback_detected": False, "details": []}
    if not stm_events: return result

    relevant_events = [e for e in stm_events if e.event_type in ["Action", "Observation"]][-10:]

    # 1. 重复动作检测
    actions = [e for e in relevant_events if e.event_type == "Action"][-5:]
    if len(actions) >= 3:
        action_signatures = [f"{','.join(sorted(act.key_entities))}_{act.content[:30]}" for act in actions]
        if action_signatures[-1] == action_signatures[-2] == action_signatures[-3]:
            result["repeated_action_detected"] = True
            result["details"].append(f"检测到连续 3 次重复动作: {actions[-1].content[:50]}")

    # 2. 无效反馈检测
    observations = [e for e in relevant_events if e.event_type == "Observation"][-5:]
    invalid_keywords = ["Permission Denied", "Timeout", "No Data Found", "Access Denied", "403", "408", "504"]
    if len(observations) >= 2:
        invalid_count = sum(
            1 for obs in observations[-2:] if any(kw.lower() in obs.content.lower() for kw in invalid_keywords))
        if invalid_count >= 2:
            result["invalid_feedback_detected"] = True
            result["details"].append("检测到连续 2 次无效工具反馈")

    return result


def _is_heuristic_pass(state: OpsAgentState) -> bool:
    """
    通过代码规则过滤掉 80% 的正常步骤，只在异常时唤醒 LLM。
    """
    messages = state.get("messages", [])
    # 找到最后一个 ToolMessage
    last_tool_msg = next((msg for msg in reversed(messages) if hasattr(msg, 'type') and msg.type == 'tool'), None)

    if not last_tool_msg:
        return False  # 没有工具执行，需要 LLM 评估

    content = last_tool_msg.content.lower()
    # 如果包含明显的错误或超时，必须走深度反思
    error_keywords = ["error", "exception", "timeout", "denied", "failed", "traceback", "oom", "crash",
                      "执行失败", "执行超时", "安全拦截", "未找到工具", "熔断", "拒绝", "不可达", "参数", "不符合"]
    if any(kw in content for kw in error_keywords):
        return False

    # 检查死胡同规则
    memory_manager = get_memory_manager()
    dead_end_info = _detect_dead_end(memory_manager.stm_events)
    if dead_end_info["repeated_action_detected"] or dead_end_info["invalid_feedback_detected"]:
        return False

    return True  # 工具执行成功且无异常，轻量级通过


def _cross_validate_reflection(reflection: ReflectionResult, state: OpsAgentState) -> bool:
    """
    Q3: 客观证据交叉验证，防止 Agent 甩锅。
    比对 LLM 的反思结论与 State 中最近的 Tool 输出。
    """
    messages = state.get("messages", [])
    recent_observations = []
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'tool':
            recent_observations.append(msg.content.lower())
            if len(recent_observations) >= 3: break

    recent_obs_text = " ".join(recent_observations)

    # 如果反思中声称“正常/无问题”，但客观证据中包含明显的错误关键词
    positive_claims = ["正常", "无异常", "没问题", "healthy", "normal", "clear"]
    negative_evidence = ["error", "timeout", "fail", "exception", "high cpu", "oom", "crash", "denied"]

    llm_text = (reflection.current_hypothesis + reflection.critique).lower()
    claim_is_positive = any(claim in llm_text for claim in positive_claims)
    evidence_is_negative = any(kw in recent_obs_text for kw in negative_evidence)

    if claim_is_positive and evidence_is_negative:
        logger.warning("[Reflection] Cross-validation failed: LLM claims normal but evidence shows errors.")
        return False  # 验证失败，存在甩锅嫌疑

    return True


async def reflect_and_adjust_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """任务 2：反思与调整节点"""

    if _is_heuristic_pass(state):
        memory_manager = get_memory_manager()
        memory_manager.add_stm("Thought", "[Heuristic Reflection] Tool executed successfully, no anomalies detected.")
        logger.info("[Reflection] Heuristic pass. Skipping LLM to save tokens.")
        return {
            "reflection_result": {"is_stuck": False, "critique": "Heuristic pass", "next_strategy": "Continue"},
            "status": AgentStatus.SUCCESS,
            "reset_count": state.get("reset_count", 0)
        }

    # === 深度反思 (LLM Reflection) ===
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "reflection", "error": "LLM missing"}]}

    memory_manager = get_memory_manager()
    stm_events = memory_manager.stm_events

    dead_end_info = _detect_dead_end(stm_events)
    dead_end_str = "\n".join(dead_end_info["details"]) if dead_end_info["details"] else "未检测到明显的重复动作或无效反馈。"
    if dead_end_info["repeated_action_detected"] or dead_end_info["invalid_feedback_detected"]:
        dead_end_str = "系统检测到异常:\n" + dead_end_str

    stm_summary = "\n".join([f"[{e.event_type}] {e.content[:200]}" for e in stm_events[-5:]])

    execution_history = state.get("execution_history", [])
    current_hypothesis = "未知"
    if execution_history:
        current_hypothesis = execution_history[-1].get("step_info", {}).get("purpose", "未知")

    evidence_summary = "\n".join([h.get("result_summary", "")[:200] for h in execution_history[-3:]])

    try:
        structured_llm = llm.with_structured_output(pydantic_model_to_json_schema(ReflectionResult), method="json_mode")
        prompt = REFLECTION_PROMPT.format(
            stm_summary=stm_summary,
            dead_end_detection=dead_end_str,
            current_hypothesis=current_hypothesis,
            evidence_summary=evidence_summary
        )

        raw_reflection = await structured_llm.ainvoke([
            SystemMessage(content="你是一个严谨的 SRE 反思专家。"),
            HumanMessage(content=prompt)
        ])

        reflection = ReflectionResult(**raw_reflection)

        if not _cross_validate_reflection(reflection, state):
            logger.warning("[Reflection] Cross-validation failed. Overriding LLM reflection to prevent blame-shifting.")
            reflection.is_stuck = True
            reflection.critique = "系统交叉验证拦截：你的反思结论与客观工具输出矛盾，存在甩锅嫌疑。强制重置策略。"
            reflection.next_strategy = "重新审查最近的工具输出，不得忽略错误日志。"

        update_state = {
            "reflection_result": reflection.model_dump(),
            "status": AgentStatus.SUCCESS
        }

        memory_manager.add_stm("Thought", f"[Reflection Critique] {reflection.critique}")

        reset_count = state.get("reset_count", 0)
        is_system_stuck = dead_end_info["repeated_action_detected"] or dead_end_info["invalid_feedback_detected"]

        if reflection.is_stuck or is_system_stuck:
            if reset_count >= 3:
                return {
                    "status": AgentStatus.FAILED,
                    "reset_count": reset_count,
                    "final_answer": "系统多次尝试后仍陷入死胡同，已触发人类介入 (HITL) 请求。",
                    "system_hints": ["触发 HITL: 连续 3 次策略重置失败"]
                }

            last_step = execution_history[-1] if execution_history else {}
            failed_action = f"{last_step.get('step_info', {}).get('action')}:{last_step.get('step_info', {}).get('target')}"
            current_failed_paths = state.get("failed_paths", [])

            update_state["current_plan"] = []
            update_state["current_step"] = 0
            update_state["reset_count"] = reset_count + 1
            update_state["system_hints"] = [f"[Reflection] 策略重置: {reflection.next_strategy}"]
            update_state["next_strategy_hint"] = reflection.next_strategy
            update_state["failed_paths"] = current_failed_paths + [failed_action]  # 追加黑名单
        else:
            update_state["reset_count"] = 0

        return update_state

    except Exception as e:
        logger.error(f"[Reflection] LLM Error: {e}")
        return {"status": AgentStatus.SUCCESS, "reset_count": state.get("reset_count", 0)}