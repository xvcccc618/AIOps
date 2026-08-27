import asyncio
import logging
from typing import List, Dict
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from rca_knowledge_base import RCACase
from reranker_service import get_reranker_service
from utils import retry_with_backoff

RERANKER_THRESHOLD = 0.6
TOP_K_AFTER_RERANK = 3

logger = logging.getLogger("RerankerNode")

#性能优化配置
MAX_CANDIDATES_TO_RERANK = 20
# 注意：这里单次超时设短一点，因为我们要重试
SINGLE_ATTEMPT_TIMEOUT = 0.8
MAX_RETRIES = 2


async def reranker_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    Reranker 节点：
    1. 动态截断候选集
    2. 使用 retry_with_backoff 处理瞬时故障
    3. 幻觉熔断
    4. 多模态数据透传
    """
    candidates = state.get("reranker_candidates", [])
    original_query = state.get("query", "")

    if not candidates:
        return {
            "historical_rca_context": "",
            "rag_status": "NO_RELEVANT_HISTORY",
            "multimodal_context": {},
            "status": AgentStatus.SUCCESS
        }

    #二次截断
    candidates_to_rerank = candidates[:MAX_CANDIDATES_TO_RERANK]

    try:
        # 调用带有重试机制的内部函数
        result = await _rerank_with_retry(original_query, candidates_to_rerank)
        return result

    except Exception as e:
        # 如果重试全部失败，执行降级
        logger.error(f"[Reranker] All retries failed. Falling back. Error: {e}", exc_info=True)
        fallback_cases = candidates[:TOP_K_AFTER_RERANK]
        context = _format_rca_context_fallback(fallback_cases)
        return {
            "historical_rca_context": context,
            "rag_status": "ERROR_FALLBACK",
            "multimodal_context": _extract_multimodal(fallback_cases),
            "status": AgentStatus.SUCCESS
        }


@retry_with_backoff(
    max_retries=MAX_RETRIES,
    base_delay=0.5,      # 初始等待 0.5 秒
    max_delay=2.0,       # 最大等待 2 秒
    exceptions=(TimeoutError, ConnectionError, Exception) # 捕获超时和网络错误
)
async def _rerank_with_retry(query: str, candidates: List[Dict]) -> dict:
    """
    内部核心逻辑，被 retry_with_backoff 装饰器包裹。
    如果此函数抛出异常，装饰器会自动进行指数退避重试。
    """
    # 在每次重试内部，依然需要设置单次超时，防止某一次请求卡死
    try:
        result = await asyncio.wait_for(
            _do_rerank_logic(query, candidates),
            timeout=SINGLE_ATTEMPT_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        # 将 asyncio 的超时转换为标准 TimeoutError，以便触发重试
        raise TimeoutError(f"Reranker attempt timed out after {SINGLE_ATTEMPT_TIMEOUT}s")


async def _do_rerank_logic(query: str, candidates: List[Dict]) -> dict:
    """
    纯粹的重排序业务逻辑，不包含重试和超时控制
    """
    service = get_reranker_service()

    # 1. 计算得分
    scores = service.compute_scores(query, candidates)

    # 2. 组装结果
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        case: RCACase = candidate["case"]
        scored_candidates.append({
            "case": case,
            "relevance_score": float(scores[i]) if i < len(scores) else 0.0
        })

    # 3. 排序
    scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 4. 业务熔断检查 (分数阈值)
    max_score = scored_candidates[0]["relevance_score"] if scored_candidates else 0

    if max_score < RERANKER_THRESHOLD:
        # 注意：分数低属于“业务成功”，不应触发重试，直接返回无结果
        logger.warning(f"[Reranker] Max score {max_score:.4f} < threshold.")
        return {
            "historical_rca_context": "",
            "rag_status": "NO_RELEVANT_HISTORY",
            "multimodal_context": {},
            "status": AgentStatus.SUCCESS
        }

    # 5. 截取 Top-K
    final_cases = scored_candidates[:TOP_K_AFTER_RERANK]
    context = _format_rca_context(final_cases)
    multimodal_ctx = _extract_multimodal(final_cases)

    return {
        "historical_rca_context": context,
        "rag_status": "RELEVANT_HISTORY_FOUND",
        "multimodal_context": multimodal_ctx,
        "reranker_debug": {"max_score": max_score},
        "status": AgentStatus.SUCCESS
    }


def _extract_multimodal(cases: List[Dict]) -> Dict:
    """提取多模态数据"""
    images = []
    graphs = []
    for item in cases:
        case: RCACase = item["case"]
        if hasattr(case, 'metrics_chart_url') and case.metrics_chart_url:
            images.append({"url": case.metrics_chart_url, "case_id": case.case_id})
        if hasattr(case, 'topology_graph_data') and case.topology_graph_data:
            graphs.append({"data": case.topology_graph_data, "case_id": case.case_id})
    return {"images": images, "graphs": graphs}


def _format_rca_context(cases: List[Dict]) -> str:
    lines = []
    for item in cases:
        case: RCACase = item["case"]
        score = item["relevance_score"]
        lines.append(
            f"- **[Score: {score:.2f}] Case {case.case_id}** (Arch: {case.architecture_version})\n"
            f"  - **Symptom**: {case.symptom}\n"
            f"  - **Root Cause**: {case.root_cause}\n"
            f"  - **Resolution**: {case.resolution}\n"
        )
    return "\n".join(lines)


def _format_rca_context_fallback(cases: List[Dict]) -> str:
    lines = []
    for item in cases:
        case: RCACase = item["case"]
        lines.append(
            f"- **[RRF Fallback] Case {case.case_id}**\n"
            f"  - **Symptom**: {case.symptom}\n"
            f"  - **Resolution**: {case.resolution}\n"
        )
    return "\n".join(lines)