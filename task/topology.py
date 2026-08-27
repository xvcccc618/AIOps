import networkx as nx
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("Topology")


class TimeAwareTopology:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._init_mock_data()

    def _init_mock_data(self):
        self.graph.add_edge(
            "Order-Service", "Payment-Service",
            relation="calls_sync", weight=0.9,
            valid_from=datetime(2023, 10, 1), valid_to=None, target_db="TiDB-B"
        )
        self.graph.add_edge(
            "Order-Service", "Payment-Service-History",
            relation="calls_sync_old", weight=0.0,
            valid_from=datetime(2023, 1, 1), valid_to=datetime(2023, 9, 30), target_db="MySQL-A"
        )
        self.graph.add_edge(
            "Payment-Service", "TiDB-B",
            relation="writes", weight=0.95,
            valid_from=datetime(2023, 10, 1), valid_to=None
        )
        self.graph.add_node("Order-Service")
        self.graph.add_node("Payment-Service")
        self.graph.add_node("TiDB-B")

    def has_node(self, node_name: str) -> bool:
        return self.graph.has_node(node_name)

    def get_neighbors(self, service: str, depth: int = 1, min_weight: float = 0.5,
                      current_time: Optional[datetime] = None) -> Dict:
        """同时计算下游 (出边) 和上游 (入边)，支持多层 BFS。"""
        if current_time is None:
            current_time = datetime.now()

        if not self.graph.has_node(service):
            return {
                "target_service": service,
                "active_dependencies": [],
                "downstream_services": [],
                "upstream_services": []
            }

        downstream_edges = []
        upstream_edges = []
        visited_nodes = {service}

        def is_valid_edge(data):
            valid_from = data.get("valid_from") or datetime.min
            valid_to = data.get("valid_to") or datetime.max
            return valid_from <= current_time <= valid_to and data.get("weight", 0) >= min_weight

        queue = [(service, 0, "both")]

        while queue:
            node, d, direction = queue.pop(0)
            if d >= depth:
                continue

            if direction in ("both", "downstream"):
                for neighbor, data in self.graph[node].items():
                    if is_valid_edge(data):
                        edge_info = {
                            "source": node, "target": neighbor,
                            "relation": data.get("relation"), "weight": data.get("weight"),
                            "meta": {k: v for k, v in data.items() if k not in ['valid_from', 'valid_to']}
                        }
                        downstream_edges.append(edge_info)
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            queue.append((neighbor, d + 1, "downstream"))

            if direction in ("both", "upstream"):
                for predecessor in self.graph.predecessors(node):
                    data = self.graph.get_edge_data(predecessor, node)
                    if data and is_valid_edge(data):
                        edge_info = {
                            "source": predecessor, "target": node,
                            "relation": data.get("relation"), "weight": data.get("weight"),
                            "meta": {k: v for k, v in data.items() if k not in ['valid_from', 'valid_to']}
                        }
                        upstream_edges.append(edge_info)
                        if predecessor not in visited_nodes:
                            visited_nodes.add(predecessor)
                            queue.append((predecessor, d + 1, "upstream"))

        return {
            "target_service": service,
            "active_dependencies": downstream_edges + upstream_edges,
            "downstream_services": list(set(e["target"] for e in downstream_edges)),
            "upstream_services": list(set(e["source"] for e in upstream_edges)),
            "downstream_paths": downstream_edges,
            "upstream_paths": upstream_edges
        }

    def get_smart_neighbors(self, service: str, depth: int = 1, min_weight: float = 0.5,
                            current_time: Optional[datetime] = None) -> Dict:
        """graph_expansion.py 调用的别名 — 与 get_neighbors 完全等价"""
        return self.get_neighbors(service, depth=depth, min_weight=min_weight, current_time=current_time)

    def add_dependency(self, source: str, target: str, relation: str = "calls",
                       weight: float = 0.6, valid_from: Optional[datetime] = None) -> None:
        """新增调用依赖边（知识反哺场景：从 RCA 报告推断出的依赖关系）。

        调用方（rca_ingestion）已校验 source/target 均为已知节点，这里不再重复校验。
        已存在的边不覆盖，保持首次录入的权重与时间窗。
        """
        if not source or not target or source == target:
            return
        if self.graph.has_edge(source, target):
            return
        self.graph.add_edge(
            source, target,
            relation=relation,
            weight=weight,
            valid_from=valid_from or datetime.now(),
            valid_to=None,
        )
        logger.info(f"[Topology] Added dependency edge: {source} -> {target} ({relation}, weight={weight})")


_topology_instance = TimeAwareTopology()


def get_topology_graph():
    return _topology_instance
