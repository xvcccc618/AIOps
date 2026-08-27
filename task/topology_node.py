import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from topology import get_topology_graph

logger = logging.getLogger("TopologyNode")


async def graph_query_node(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    拓扑查询节点：
    1. 从 State 中提取核心实体 (service_name)
    2. 查询拓扑图谱获取上下游依赖
    3. 将结构化信息转化为文本上下文，写入 state['topology_context']
       注意：此处使用自然语言描述而非 Mermaid/JSON，以优化 LLM 理解效率并减少 Token 消耗。
    """

    # 1. 提取实体
    entities = state.get("extracted_entities", {})
    query = state.get("query", "")

    # 尝试从提取的实体中获取服务名，如果没有则尝试从 query 中简单匹配 (降级)
    service_name = entities.get("service_name") or entities.get("SERVICE_NAME")

    if not service_name:
        logger.info("[Topology] No specific service entity found in extracted_entities. Skipping topology lookup.")
        return {
            "topology_context": "",
            "status": AgentStatus.SUCCESS
        }

    try:
        # 2. 调用拓扑图
        topo_graph = get_topology_graph()
        topo_data = topo_graph.get_neighbors(service_name, depth=2)

        if "error" in topo_data:
            logger.warning(f"[Topology] Error fetching neighbors for {service_name}: {topo_data['error']}")
            return {
                "topology_context": "",
                "status": AgentStatus.SUCCESS
            }

        # 3. 格式化上下文
        context_str = _format_topology_context(topo_data)

        logger.info(f"[Topology] Successfully generated context for {service_name}")

        return {
            "topology_context": context_str,
            "status": AgentStatus.SUCCESS
        }

    except Exception as e:
        logger.error(f"[Topology] Critical error: {e}", exc_info=True)
        # 降级：返回空上下文，不阻断流程
        return {
            "topology_context": "",
            "status": AgentStatus.SUCCESS
        }


def _format_topology_context(data: Dict[str, Any]) -> str:
    """
    将拓扑数据格式化为 Prompt 友好的自然语言文本
    """
    target = data.get("target_service", "Unknown")
    upstream = data.get("upstream_services", [])
    downstream = data.get("downstream_services", [])
    up_paths = data.get("upstream_paths", [])
    down_paths = data.get("downstream_paths", [])

    lines = []
    lines.append(f"【系统全局拓扑参考】")
    lines.append(f"当前排查目标: {target}")

    # 上游
    if upstream:
        lines.append(f"上游依赖 (调用方): {', '.join(upstream)}")
        for p in up_paths:
            lines.append(f"  - {p['source']} --[{p['relation']}]--> {p['target']}")
    else:
        lines.append("上游依赖: 无直接上游或未知")

    # 下游
    if downstream:
        lines.append(f"下游依赖 (被调用方/存储): {', '.join(downstream)}")
        for p in down_paths:
            lines.append(f"  - {p['source']} --[{p['relation']}]--> {p['target']}")
    else:
        lines.append("下游依赖: 无直接下游")

    lines.append("")
    lines.append("【排查指导】：")
    lines.append(f"在分析 {target} 故障时，请务必考虑其下游依赖 (如超时、连接池耗尽) 或上游流量突发引发级联雪崩的可能性。")

    return "\n".join(lines)