"""
真实端点冒烟验证：LLM / Embedding / Milvus / Rerank / 完整 RAG 节点
用法：python verify_real.py   （需 .env 已配置 OPENAI_API_KEY 与 SILICONFLOW_API_KEY）
全部通过输出 ALL REAL ENDPOINTS VERIFIED，任一失败退出码为 1。
"""
import asyncio
import sys
import time

import settings  # noqa: F401  先触发 .env 加载

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    (PASSED if ok else FAILED).append(name)


async def main():
    # 1. LLM 真实调用
    try:
        from main import create_llm_instance
        from langchain_core.messages import HumanMessage
        llm = create_llm_instance()
        t0 = time.time()
        resp = await llm.ainvoke([HumanMessage(content="用一句话回答：1+1等于？")])
        cost = time.time() - t0
        check("LLM 真实调用", bool(resp.content.strip()),
              f"model 返回 {len(resp.content)} 字符, {cost:.1f}s")
    except Exception as e:
        check("LLM 真实调用", False, f"{type(e).__name__}: {str(e)[:150]}")

    # 2. 远程 Embedding（bge-m3）
    kb = None
    try:
        from rca_knowledge_base import get_rca_kb
        kb = get_rca_kb()  # 初始化时会把样例案例 embed 写入 Milvus
        vecs = kb.embedder.embed(["订单服务频繁重启"])
        ok = vecs is not None and len(vecs) == 1 and len(vecs[0]) > 0
        check("远程 Embedding(bge-m3)", ok,
              f"维度={len(vecs[0]) if vecs else 0}")
    except Exception as e:
        check("远程 Embedding(bge-m3)", False, f"{type(e).__name__}: {str(e)[:150]}")

    # 3. Milvus 写入与检索
    try:
        if kb and kb.milvus_client:
            hits = kb.search_with_parent_mapping("订单服务 Pod 频繁重启 OOM 堆内存溢出", k=5)
            ok = len(hits) > 0 and hits[0]["score"] > 0
            top = hits[0] if hits else {}
            check("Milvus 检索", ok,
                  f"{len(hits)} 条命中, top_score={top.get('score', 0):.4f}, case={top.get('doc', {}).get('metadata', {}).get('case_id')}")
        else:
            check("Milvus 检索", False, "client 不可用")
    except Exception as e:
        check("Milvus 检索", False, f"{type(e).__name__}: {str(e)[:150]}")

    # 4. BGE 重排（真实 API）
    try:
        from reranker_service import get_reranker_service
        svc = get_reranker_service()
        scores = svc.compute_scores(
            "订单服务 Pod CrashLoopBackOff OOM",
            [{"text": "订单服务 Pod 频繁重启，OutOfMemoryError: Java heap space，本地缓存未设上限"},
             {"text": "网关 P99 延迟上升，Full GC STW 拖垮上游"}],
        )
        ok = (not svc.is_mock) and len(scores) == 2 and scores[0] > scores[1]
        check("BGE 重排(真实API)", ok,
              f"scores={[round(s, 4) for s in scores]}, mock={svc.is_mock}")
    except Exception as e:
        check("BGE 重排(真实API)", False, f"{type(e).__name__}: {str(e)[:150]}")

    # 5. 完整 RAG 节点（双路召回→RRF→重排→预算装配）
    try:
        from bone import IncidentSeverity
        from retrieve import rag_retrieval_node
        state = {
            "query": "订单服务 Pod 频繁重启 CrashLoopBackOff OOM",
            "incident_severity": IncidentSeverity.P2,
            "related_components_for_filter": [],
        }
        result = await rag_retrieval_node(state, {"configurable": {}})
        ctx = result.get("retrieved_context", [])
        ok = result.get("status") and len(ctx) > 0 and len(ctx[0]) > 50
        check("完整 RAG 节点", ok,
              f"rag_status={result.get('rag_status')}, 上下文长度={len(ctx[0]) if ctx else 0} 字符")
    except Exception as e:
        check("完整 RAG 节点", False, f"{type(e).__name__}: {str(e)[:150]}")

    print("\n" + "=" * 50)
    if FAILED:
        print(f"FAILED: {len(FAILED)} 项未通过 -> {FAILED}")
        sys.exit(1)
    print(f"ALL REAL ENDPOINTS VERIFIED ({len(PASSED)}/{len(PASSED)})")


if __name__ == "__main__":
    asyncio.run(main())
