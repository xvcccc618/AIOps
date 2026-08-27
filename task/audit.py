import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus

logger = logging.getLogger("AuditNode")

HIGH_RISK_ACTIONS = {"delete", "drop", "truncate", "restart", "kill", "format"}


async def audit_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    审计节点（接在 RAG 检索之后、生成/规划之前）：
    1. 检测检索回来的历史案例方案是否包含高危操作，防止"知识库投毒"。
    2. 【注意】当前版本仅支持静态关键词匹配，动态监控交叉验证需补充 metrics_collector_node。
    读取字段为新版 RAG 链路的 retrieved_context。
    """
    rag_status = state.get("rag_status", "")
    retrieved_context = state.get("retrieved_context", []) or []

    # 没有检索到相关历史（或流程未命中历史分支）时无需审计
    if rag_status != "RELEVANT_HISTORY_FOUND" or not retrieved_context:
        return {"audit_status": "SAFE", "status": AgentStatus.SUCCESS}

    # retrieved_context 是 List[str]（每段一个文档/预算块）
    historical_context = "\n".join(retrieved_context).lower()

    risk_level = "LOW"
    warnings = []

    # 1. 静态规则检测：高危关键词
    if any(action in historical_context for action in HIGH_RISK_ACTIONS):
        risk_level = "HIGH"
        warnings.append("Detected high-risk operations in historical context (e.g., restart, delete).")

    # 2. 动态交叉验证 (已禁用，因为 State 中缺乏 current_metrics_summary)
    # TODO: 在此处添加 metrics_collector_node 的输出读取逻辑

    if risk_level in ["HIGH", "CRITICAL"]:
        logger.warning(f"[Audit] Risk Level: {risk_level}. Warnings: {warnings}")
        return {
            "audit_status": "RISK_DETECTED",
            "audit_warnings": warnings,
            "risk_level": risk_level,
            "system_hints": ["[审计] 历史案例方案含高危操作关键词，生成/执行前请人工复核。"],
            "status": AgentStatus.SUCCESS
        }

    return {
        "audit_status": "SAFE",
        "status": AgentStatus.SUCCESS
    }
