import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from rca_schema import RCAResult
from rca_knowledge_base import get_rca_kb, RCACase
from topology import get_topology_graph
from utils import retry_with_backoff, get_circuit_breaker, to_structured_output_schema
from memory_manager import get_memory_manager

logger = logging.getLogger("RCAIngestionNode")

# 配置常量
LLM_TIMEOUT_SECONDS = 30.0
VECTOR_DB_TIMEOUT_SECONDS = 10.0
TOPOLOGY_EXTRACTION_PROMPT = """你是一个运维架构专家。请从以下故障分析文本中提取微服务/组件之间的明确调用依赖关系。
只提取文本中明确提到或强烈暗示的调用关系（如 A 调用 B，A 依赖 B，A 请求 B，A 写入 B）。
不要猜测或推断隐含关系。

输出严格的 JSON 数组格式：[{"source": "ServiceA", "target": "ServiceB"}]
如果没有明确关系，返回空数组 []。

文本：
{text}
"""


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=5.0, exceptions=(Exception,))
async def _write_to_vector_db(kb, new_case: RCACase):
    """
    内部函数：执行实际的向量库写入操作，支持重试。
    """
    kb.add_case(new_case)


async def ingest_to_vector_kb(rca_data: Dict[str, Any]) -> bool:
    try:
        kb = get_rca_kb()
        rca = RCAResult(**rca_data)
        resolution_steps = "\n".join([f"{item.type.value}: {item.description}" for item in rca.action_items])

        new_case = RCACase(
            case_id=f"RCA-AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            symptom=rca.incident_summary,
            root_cause=rca.root_cause_analysis,
            resolution=resolution_steps,
            architecture_version="current",
            is_deprecated=False,
            metrics_chart_url=None,
            topology_graph_data=None
        )

        # 使用超时包装写入操作
        await asyncio.wait_for(
            _write_to_vector_db(kb, new_case),
            timeout=VECTOR_DB_TIMEOUT_SECONDS
        )

        logger.info(f"[Ingestion] Successfully added case {new_case.case_id} to Vector KB.")
        return True
    except asyncio.TimeoutError:
        logger.error(f"[Ingestion] Vector DB write timed out after {VECTOR_DB_TIMEOUT_SECONDS}s")
        return False
    except Exception as e:
        # 重试装饰器会在内部处理重试，如果这里捕获到异常，说明重试已全部失败
        logger.error(f"[Ingestion] Failed to ingest to Vector KB after retries: {e}", exc_info=True)
        return False


async def update_topology_graph(rca_data: Dict[str, Any], llm_instance=None) -> bool:
    tool_name = "rca_topology_llm"
    cb = get_circuit_breaker(tool_name)

    # 1. 检查熔断器状态
    if cb.is_open():
        logger.warning(f"[Ingestion] Circuit breaker for '{tool_name}' is OPEN. Skipping LLM call.")
        return True  # 视为成功（降级处理），不阻断主流程

    try:
        rca = RCAResult(**rca_data)
        topo_graph = get_topology_graph()

        text_to_analyze = f"{rca_data.get('root_cause_analysis', '')} {' '.join([ev.get('content', '') for ev in rca_data.get('evidence_chain', [])])}"

        inferred_dependencies = []

        if llm_instance:
            try:
                structured_llm = llm_instance.with_structured_output(to_structured_output_schema({
                    "dependencies": List[Dict[str, str]]
                }), method="json_mode")

                # 2. 带超时的 LLM 调用
                result = await asyncio.wait_for(
                    structured_llm.ainvoke([
                        {"role": "system", "content": TOPOLOGY_EXTRACTION_PROMPT.format(text=text_to_analyze[:2000])},
                        {"role": "human", "content": "请提取依赖关系。"}
                    ]),
                    timeout=LLM_TIMEOUT_SECONDS
                )

                # 调用成功，重置熔断器计数
                cb.record_success()
                inferred_dependencies = result.get("dependencies", [])

            except asyncio.TimeoutError:
                logger.warning(f"[Ingestion] LLM extraction timed out after {LLM_TIMEOUT_SECONDS}s")
                cb.record_failure()  # 记录失败
                return True  # 降级：跳过更新
            except Exception as e:
                logger.warning(f"[Ingestion] LLM extraction failed: {e}")
                cb.record_failure()  # 记录失败
                return True  # 降级：跳过更新
        else:
            logger.warning("[Ingestion] No LLM provided for topology inference. Skipping.")
            return True

        # 校验并写入拓扑图
        known_nodes = set(topo_graph.graph.nodes())
        added_count = 0

        for dep in inferred_dependencies:
            src = dep.get("source")
            tgt = dep.get("target")

            # 严格校验：源和目标必须都是已知的有效节点
            if src and tgt and src in known_nodes and tgt in known_nodes and src != tgt:
                if not topo_graph.graph.has_edge(src, tgt):
                    topo_graph.add_dependency(src, tgt, relation="inferred_from_rca", weight=0.6)
                    logger.info(f"[Ingestion] Inferred & Validated dependency: {src} -> {tgt}")
                    added_count += 1
            else:
                logger.debug(f"[Ingestion] Rejected invalid/hallucinated dependency: {src} -> {tgt}")

        logger.info(f"[Ingestion] Topology graph update complete. Added {added_count} valid edges.")
        return True

    except Exception as e:
        # 顶层异常捕获，防止未预见的错误导致节点崩溃
        logger.error(f"[Ingestion] Unexpected error in topology update: {e}", exc_info=True)
        cb.record_failure()
        return True  # 即使出错也返回 True，避免标记整个 ingestion 为失败，除非业务强依赖此步骤


async def knowledge_ingestion_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    rca_json = state.get("rca_report_json")
    if not rca_json:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "ingestion", "error": "No RCA report found"}]}

    logger.info("[Ingestion] Starting knowledge feedback loop with resilience patterns...")

    # 并行执行两个独立的任务
    vec_success, topo_success = await asyncio.gather(
        ingest_to_vector_kb(rca_json),
        update_topology_graph(rca_json, llm_instance=config.get("configurable", {}).get("llm_instance"))
    )

    ingestion_status = {
        "vector_kb_updated": vec_success,
        "topology_graph_updated": topo_success,
        "timestamp": datetime.now().isoformat()
    }

    # 如果关键的知识库写入失败，可以考虑标记部分失败，但通常我们允许部分成功
    # 长期经验提炼（LTM）：从本次 RCA 提取经验存入长期记忆，失败不影响主流程
    ltm_success = False
    try:
        caller_namespace = state.get("extracted_entities", {}).get("tenant_id", "default_tenant")
        get_memory_manager().extract_and_save_experience(state, rca_json, caller_namespace)
        ltm_success = True
    except Exception as e:
        logger.error(f"[Ingestion] LTM experience extraction failed: {e}")
    ingestion_status["ltm_experience_saved"] = ltm_success

    final_status = AgentStatus.SUCCESS
    hints = ["[System] RCA 报告已自动回流至知识库。"]

    if not vec_success:
        hints.append("[Warning] 向量知识库更新失败，可能影响后续相似故障检索。")
    if not topo_success:
        # 注意：update_topology_graph 在降级时也会返回 True，所以这里主要捕获显式失败
        pass

    logger.info(f"[Ingestion] Completed. Vector: {vec_success}, Topo: {topo_success}")

    return {
        "rca_ingestion_status": ingestion_status,
        "status": final_status,
        "system_hints": hints
    }
