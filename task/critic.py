import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from utils import to_structured_output_schema

logger = logging.getLogger("CriticNode")

CRITIC_SYSTEM_PROMPT = """你是一个高级 SRE 仲裁专家。你的任务是评估两种不同来源的故障分析线索，并给出最终建议。

【输入信息】
1. **Vector RAG 结论** (基于历史知识库):
   {vector_rca}

2. **Graph RAG 结论** (基于实时拓扑与依赖):
   {graph_topology_context}

【裁决规则】
1. **时效性优先**：如果 Vector RAG 提到的案例超过 6 个月，而 Graph RAG 显示下游有实时异常（如超时、错误率飙升），优先相信 Graph RAG。
2. **特异性优先**：如果 Vector RAG 明确指出是“代码逻辑 Bug”且症状完全匹配（如特定报错堆栈），而 Graph RAG 仅显示一般性依赖，优先相信 Vector RAG。
3. **级联效应**：如果 Graph RAG 显示核心依赖（如 DB, Redis）存在普遍性延迟，即使 Vector RAG 指向代码 Bug，也应首先排查基础设施，因为基础设施问题常表现为应用层异常。

【输出要求】
请以 JSON 格式输出：
{
    "final_verdict": "VECTOR_RAG" | "GRAPH_RAG" | "HYBRID",
    "confidence": 0-1,
    "reasoning": "详细的裁决理由",
    "recommended_action": "具体的下一步排查建议"
}
"""


async def critic_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    裁决节点：对比 Vector RAG 和 Graph RAG 的结果
    """
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "critic", "error": "LLM missing"}]}

    # 获取 Vector RAG 的结果 (假设存储在 retrieved_context 或 final_answer 的中间状态)
    # 这里我们假设 retrieved_context 中包含了最相关的历史 RCA 摘要
    vector_context = "\n".join(state.get("retrieved_context", [])[:3])  # 取前3条最相关

    # 获取 Graph RAG 的结果
    graph_context = state.get("topology_context", "")

    if not vector_context and not graph_context:
        return {"status": AgentStatus.SUCCESS, "critic_decision": "NO_DATA"}

    try:
        structured_llm = llm.with_structured_output(to_structured_output_schema({
            "final_verdict": str,
            "confidence": float,
            "reasoning": str,
            "recommended_action": str
        }), method="json_mode")

        response = await structured_llm.ainvoke([
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT.format(
                vector_rca=vector_context if vector_context else "无历史匹配案例",
                graph_topology_context=graph_context if graph_context else "无拓扑异常"
            )},
            {"role": "human", "content": "请进行裁决。"}
        ])

        logger.info(f"[Critic] Verdict: {response['final_verdict']}, Reason: {response['reasoning']}")

        return {
            "critic_decision": response,
            "system_hints": [
                f"[Critic] 裁决结果: {response['final_verdict']}. 建议: {response['recommended_action']}"],
            "status": AgentStatus.SUCCESS
        }

    except Exception as e:
        logger.error(f"[Critic] Error: {e}")
        return {
            "status": AgentStatus.FAILED,
            "error_log": [{"node": "critic", "error": str(e)}],
            "system_hints": ["⚠️ [Critic] 裁决失败，默认采用混合策略"]
        }