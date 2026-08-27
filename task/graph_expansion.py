import logging
from typing import List, Dict, Any, Tuple
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from topology import get_topology_graph
from utils import to_structured_output_schema

logger = logging.getLogger("GraphExpansionNode")

GRAPH_EXPANSION_PROMPT = """你是一个运维领域实体识别专家。请从用户查询中提取关键业务服务或组件实体。
如果查询中包含模糊的业务术语（如“结账”、“下单”），请将其映射到具体的微服务名称。

输出格式为 JSON：
{
    "entities": [
        {"original_term": "结账", "mapped_service": "Payment-Service", "confidence_reason": "业务逻辑强相关"}
    ]
}
"""


async def graph_expansion_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    llm = config.get("configurable", {}).get("llm_instance")
    if not llm:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "graph_expansion", "error": "LLM missing"}]}

    query = state.get("query", "")
    topology = get_topology_graph()

    # 1. 实体提取
    try:
        structured_llm = llm.with_structured_output(to_structured_output_schema({
            "entities": List[Dict[str, str]]
        }), method="json_mode")
        result = await structured_llm.ainvoke([
            {"role": "system", "content": GRAPH_EXPANSION_PROMPT},
            {"role": "human", "content": query}
        ])
        extracted_entities = result.get("entities", [])
        logger.info(f"[GraphExpansion] Raw Extracted Entities: {extracted_entities}")
    except Exception as e:
        logger.error(f"[GraphExpansion] Entity Extraction Failed: {e}")
        extracted_entities = []

    # 2. 置信度校验与过滤
    valid_entities = []
    invalid_mappings = []

    for item in extracted_entities:
        original_term = item.get("original_term", "")
        mapped_service = item.get("mapped_service", "")

        # 校验 1: 拓扑存在性校验
        # 检查 mapped_service 是否是图中的有效节点
        is_valid_node = topology.graph.has_node(mapped_service)

        # 校验 2: 简单启发式置信度 (实际生产中可结合 LLM self-consistency)
        # 如果 LLM 给出的理由很短或包含不确定词汇，降低置信度
        reason = item.get("confidence_reason", "")
        low_confidence_keywords = ["可能", "猜测", "不确定", "maybe", "guess"]
        is_low_confidence = any(kw in reason.lower() for kw in low_confidence_keywords)

        if is_valid_node and not is_low_confidence:
            valid_entities.append(mapped_service)
            logger.info(f"[Validation] Accepted: '{original_term}' -> '{mapped_service}'")
        else:
            invalid_mappings.append(original_term)
            logger.warning(
                f"[Validation] Rejected: '{original_term}' -> '{mapped_service}' (Valid Node: {is_valid_node}, Low Conf: {is_low_confidence})")

    # 3. 安全回退策略
    # 如果所有映射都失败，或者 valid_entities 为空，则保留原始查询中的关键词
    expansion_queries = [query]  # 始终保留原始 Query 作为基准

    related_components = set()

    if valid_entities:
        for entity in valid_entities:
            # 获取下游依赖
            neighbors = topology.get_smart_neighbors(entity, depth=1, min_weight=0.6)
            downstream = neighbors.get("downstream_services", [])
            related_components.update(downstream)

            # 构建技术关键词扩展 Query
            if downstream:
                tech_keywords = " OR ".join(downstream)
                # 使用括号确保优先级，避免语义漂移
                expanded_query = f"{query} ({tech_keywords} latency OR error OR timeout)"
                expansion_queries.append(expanded_query)

            # 添加针对该实体的通用故障词
            expansion_queries.append(f"{entity} high latency connection pool deadlock thread stuck")
    else:
        # 回退：如果没有有效映射，尝试从原始 query 中提取名词作为备选，或者直接依赖原始 query
        logger.info("[Fallback] No valid entities mapped. Using original query only.")
        pass

    # 去重
    expansion_queries = list(set(expansion_queries))

    logger.info(f"[GraphExpansion] Final Expanded Queries: {expansion_queries}")
    logger.info(f"[GraphExpansion] Related Components for Metadata Filter: {related_components}")

    return {
        "expanded_search_queries": expansion_queries,
        "related_components_for_filter": list(related_components),
        "valid_mapped_entities": valid_entities,  # 传递给后续节点用于提示
        "status": AgentStatus.SUCCESS
    }