"""
运维监控 MCP Tools（真实后端版）
封装 Prometheus / K8s 查询为 MCP 工具。后端不可达时返回降级信息，不抛异常。
"""
import asyncio
from typing import Any, Dict
import logging

from mcp_server import mcp_server
from backends import get_prometheus_backend, get_k8s_backend

logger = logging.getLogger("MCPToolsMonitor")


async def query_prometheus_metrics(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """查询 Prometheus 监控指标"""
    service = arguments.get("service", "unknown")
    metric_type = arguments.get("metric_type", "all")
    duration_minutes = arguments.get("duration_minutes", 15)

    prom = get_prometheus_backend()
    if not await prom.is_healthy():
        return {
            "service": service,
            "status": "UNAVAILABLE",
            "message": f"Prometheus 不可达（{prom.base_url}），无法查询指标。请确认服务已启动。"
        }

    # 按 metric_type 选择 PromQL（cadvisor / node exporter 常见指标）
    label = f'pod=~"{service}.*"'
    promql_map = {
        "cpu": f'sum(rate(container_cpu_usage_seconds_total{{{label}}}[5m]))',
        "memory": f'sum(container_memory_working_set_bytes{{{label}}})',
        "network": f'sum(rate(container_network_receive_bytes_total{{{label}}}[5m]))',
        "disk": 'sum(node_filesystem_avail_bytes{mountpoint="/"})',
    }

    metrics: Dict[str, Any] = {
        "service": service,
        "time_range": f"last {duration_minutes} minutes",
        "status": "OK",
    }

    types = ["cpu", "memory", "network", "disk"] if metric_type == "all" else [metric_type]
    for mt in types:
        if mt not in promql_map:
            continue
        result = await prom.query(promql_map[mt])
        if "error" in result:
            metrics[mt] = {"error": result["error"]}
            continue
        data = result.get("data", [])
        if data:
            value = float(data[0]["value"][1]) if data[0].get("value") else 0.0
            metrics[mt] = {"value": round(value, 4), "unit": _metric_unit(mt)}
        else:
            metrics[mt] = {"value": None, "note": "no data (服务可能无对应容器指标)"}

    return metrics


def _metric_unit(metric_type: str) -> str:
    return {"cpu": "cores", "memory": "bytes", "network": "bytes/s", "disk": "bytes"}.get(metric_type, "")


async def fetch_k8s_pod_logs(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """获取 Kubernetes Pod 日志"""
    namespace = arguments.get("namespace", "default")
    pod_name = arguments.get("pod_name")
    container = arguments.get("container")
    lines = arguments.get("lines", 100)

    if not pod_name:
        raise ValueError("pod_name is required")

    k8s = get_k8s_backend()
    if not k8s.available:
        return {
            "namespace": namespace, "pod": pod_name,
            "status": "UNAVAILABLE",
            "message": f"⚠️ K8s 集群不可达: {k8s._load_error}"
        }

    result = await k8s.get_pod_logs(namespace=namespace, pod_name=pod_name,
                                     tail_lines=lines, container=container)
    if "error" in result:
        return {"namespace": namespace, "pod": pod_name, "error": result["error"]}

    log_lines = result["logs"].split("\n") if result["logs"] else []
    return {
        "namespace": namespace,
        "pod": pod_name,
        "container": container,
        "line_count": len(log_lines),
        "logs": result["logs"][:5000],
    }


async def get_k8s_pod_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """获取 Kubernetes Pod 状态"""
    namespace = arguments.get("namespace", "default")
    pod_name = arguments.get("pod_name")

    if not pod_name:
        raise ValueError("pod_name is required")

    k8s = get_k8s_backend()
    if not k8s.available:
        return {
            "namespace": namespace, "pod_name": pod_name,
            "status": "UNAVAILABLE",
            "message": f"⚠️ K8s 集群不可达: {k8s._load_error}"
        }

    result = await k8s.describe_pod(namespace=namespace, pod_name=pod_name)
    if "error" in result:
        return {"namespace": namespace, "pod_name": pod_name, "error": result["error"]}

    return {
        "namespace": namespace,
        "pod_name": pod_name,
        "status": result.get("status"),
        "restarts": result.get("restarts", 0),
        "node": result.get("node"),
        "ip": result.get("ip"),
        "last_state": result.get("last_state", {}),
    }


async def query_service_dependencies(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """查询服务依赖关系（从拓扑图获取，非 mock 硬编码）"""
    from topology import get_topology_graph

    service = arguments.get("service")
    if not service:
        raise ValueError("service is required")

    topo = get_topology_graph()
    neighbors = topo.get_neighbors(service, depth=1)
    return {
        "service": service,
        "dependencies": {
            "upstream": neighbors.get("upstream_services", []),
            "downstream": neighbors.get("downstream_services", []),
        },
    }


# 注册监控相关工具到 MCP Server
def register_monitor_tools():
    """注册所有监控工具"""

    mcp_server.register_tool(
        name="query_prometheus_metrics",
        description="查询 Prometheus 监控指标（CPU、内存、磁盘、网络）",
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "服务名称"},
                "metric_type": {
                    "type": "string",
                    "enum": ["cpu", "memory", "disk", "network", "all"],
                    "description": "指标类型"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "查询时间范围（分钟）",
                    "default": 15
                }
            },
            "required": ["service"]
        },
        handler=query_prometheus_metrics,
        timeout_seconds=10
    )

    mcp_server.register_tool(
        name="fetch_k8s_pod_logs",
        description="获取 Kubernetes Pod 日志",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "default"},
                "pod_name": {"type": "string", "description": "Pod 名称"},
                "container": {"type": "string", "default": "main"},
                "lines": {"type": "integer", "default": 100},
            },
            "required": ["pod_name"]
        },
        handler=fetch_k8s_pod_logs,
        timeout_seconds=15
    )

    mcp_server.register_tool(
        name="get_k8s_pod_status",
        description="获取 Kubernetes Pod 状态信息",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "default"},
                "pod_name": {"type": "string", "description": "Pod 名称"}
            },
            "required": ["pod_name"]
        },
        handler=get_k8s_pod_status,
        timeout_seconds=10
    )

    mcp_server.register_tool(
        name="query_service_dependencies",
        description="查询服务的上下游依赖关系",
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "服务名称"}
            },
            "required": ["service"]
        },
        handler=query_service_dependencies,
        timeout_seconds=10
    )

    logger.info("Registered 4 monitoring tools to MCP Server")


# 自动注册工具
register_monitor_tools()
