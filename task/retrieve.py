import re
import logging
from typing import List, Dict, Set, Tuple
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus, IncidentSeverity
from rca_knowledge_base import get_rca_kb
from reranker_service import get_reranker_service
from topology import get_topology_graph
from token_budget_manager import TokenBudgetManager

logger = logging.getLogger("RAGRetrievalNode")

TOP_K_PER_PATH = 10
K_RRF = 60
# 扩展查询最多额外召回几路（不含原始 query 路），控制远程 Embedding 调用次数
MAX_EXTRA_EXPANSION_PATHS = 2


async def rag_retrieval_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    高级 RAG 检索节点：多路召回（语义 + 扩展查询 + 图引导过滤）→ RRF → 重排 → 智能预算装配。
    """
    query = state.get("query", "")
    related_components = state.get("related_components_for_filter", [])
    expanded_queries = state.get("expanded_search_queries", [])
    severity = state.get("incident_severity", IncidentSeverity.P3)

    kb = get_rca_kb()
    reranker = get_reranker_service()

    if not query:
        return {"retrieved_context": [], "status": AgentStatus.FAILED}

    paths = []

    # --- 路径 A：原始 query 纯语义召回 ---
    semantic_results = kb.search_with_parent_mapping(query, k=TOP_K_PER_PATH)
    if semantic_results:
        paths.append(semantic_results)

    # --- 路径 C：扩展查询召回（消费 graph_expansion 产出的 expanded_search_queries）---
    extra_queries = [q for q in expanded_queries if q and q != query][:MAX_EXTRA_EXPANSION_PATHS]
    for eq in extra_queries:
        expansion_results = kb.search_with_parent_mapping(eq, k=TOP_K_PER_PATH)
        if expansion_results:
            paths.append(expansion_results)
            logger.info(f"[RAG] Expansion path recall: {eq[:60]}... -> {len(expansion_results)} hits")

    # --- 路径 B：图引导过滤召回（真实布尔过滤）---
    if related_components:
        graph_guided_results = kb.search_with_parent_mapping_and_filter(
            query, allowed_components=related_components, k=TOP_K_PER_PATH
        )
        if graph_guided_results:
            paths.append(graph_guided_results)
            logger.info(f"[RAG] Graph-guided path recall: components={related_components} -> {len(graph_guided_results)} hits")

    if not paths:
        return {"retrieved_context": [], "rag_status": "NO_RELEVANT_HISTORY", "status": AgentStatus.SUCCESS}

    fused_results = reciprocal_rank_fusion(paths, k=K_RRF)

    candidates_for_rerank = fused_results[:20]
    rerank_inputs = [{"text": res['doc']['content']} for res in candidates_for_rerank]
    rerank_scores = reranker.compute_scores(query, rerank_inputs)

    scored_results = []
    for i, res in enumerate(candidates_for_rerank):
        res['rerank_score'] = rerank_scores[i] if i < len(rerank_scores) else 0.0
        scored_results.append(res)

    scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_k_results = scored_results[:5]

    if not top_k_results:
        return {"retrieved_context": [], "rag_status": "NO_RELEVANT_HISTORY", "status": AgentStatus.SUCCESS}

    # --- Step 4: Prepare Data for Smart Budgeting ---
    # 构建结构化数据：包含子块内容、父块全文、得分
    processed_items = []
    seen_parents = set()
    for res in top_k_results:
        p_hash = res['parent_hash']
        child_content = res['doc']['content']
        score = res['rerank_score']

        # 获取父块文本
        parent_text = kb.get_parent_text(p_hash)

        # 去重：如果同一个父块已经被处理过，跳过（或者合并得分，这里简单跳过）
        if p_hash in seen_parents:
            continue

        processed_items.append({
            "child_content": child_content,
            "parent_text": parent_text,
            "score": score,
            "p_hash": p_hash
        })
        seen_parents.add(p_hash)

    # --- Step 5: Apply Weighted Token Budget with Smart Extension ---
    retrieved_context_parts = []

    if processed_items:
        retrieved_context_parts = TokenBudgetManager.apply_budget_to_rag_context(
            raw_items=processed_items,
            severity=severity
        )

    retrieved_context = "\n\n---\n\n".join(retrieved_context_parts)

    logger.info(f"[RAG] Final Context Generated. Parts: {len(retrieved_context_parts)}, Len: {len(retrieved_context)}")

    return {
        "retrieved_context": [retrieved_context] if retrieved_context else [],
        "rag_status": "RELEVANT_HISTORY_FOUND" if retrieved_context else "NO_RELEVANT_HISTORY",
        "status": AgentStatus.SUCCESS
    }


def reciprocal_rank_fusion(scored_docs_list: List[List[Dict]], k: int = K_RRF) -> List[Dict]:
    fused_scores = {}
    for docs in scored_docs_list:
        for rank, item in enumerate(docs):
            doc_id = item["doc"].get("id")
            if not doc_id: continue
            rrf_score = 1.0 / (k + rank)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "doc": item["doc"],
                    "rrf_score": 0.0,
                    "parent_hash": item.get("parent_hash"),
                    "original_scores": []
                }
            fused_scores[doc_id]["rrf_score"] += rrf_score
            fused_scores[doc_id]["original_scores"].append(item["score"])
    return sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
