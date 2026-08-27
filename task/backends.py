"""
真实后端适配层：MySQL / Prometheus / K8s 三个真实数据源的统一客户端。
设计原则：
1. 连接失败不抛给上层 —— 返回结构化错误信息，工具层转成 ToolMessage，链路不中断
2. 所有网络调用带显式超时
3. 懒加载：首次调用才建连，失败不阻塞进程启动
"""
import asyncio
import logging
import time
from typing import Optional, List, Dict, Any

import pymysql

from settings import get_db_config, get_prometheus_config, get_k8s_config

logger = logging.getLogger("Backends")

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 10
PROM_TIMEOUT_SECONDS = 8


# ================= MySQL =================

class MySQLBackend:
    """MySQL 真实后端：慢查询、连接池、健康检查、只读 SQL"""

    def __init__(self):
        self._cfg = get_db_config()

    def _connect(self) -> pymysql.connections.Connection:
        return pymysql.connect(
            host=self._cfg["host"],
            port=self._cfg["port"],
            user=self._cfg["user"],
            password=self._cfg["password"],
            database=self._cfg["database"],
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=QUERY_TIMEOUT_SECONDS,
            charset="utf8mb4",
        )

    async def query(self, sql: str, params=None, fetch_all: bool = True) -> Dict[str, Any]:
        """执行查询，返回 {"columns": [...], "rows": [...]} 或 {"error": ...}"""
        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    columns = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall() if fetch_all else []
                    return {"columns": columns, "rows": [list(r) for r in rows]}
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"MySQL query failed: {type(e).__name__}: {str(e)[:200]}")
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    async def slow_queries(self, threshold_ms: int = 1000, limit: int = 10) -> Dict[str, Any]:
        """从 performance_schema 取慢查询摘要（按平均耗时排序）"""
        sql = """
            SELECT DIGEST_TEXT AS sql_text,
                   COUNT_STAR AS exec_count,
                   ROUND(AVG_TIMER_WAIT / 1e9, 1) AS avg_ms,
                   ROUND(MAX_TIMER_WAIT / 1e9, 1) AS max_ms,
                   SUM_ROWS_EXAMINED AS rows_examined
            FROM performance_schema.events_statements_summary_by_digest
            WHERE DIGEST_TEXT IS NOT NULL
            ORDER BY AVG_TIMER_WAIT DESC
            LIMIT %s
        """
        result = await self.query(sql, (limit,))
        if "error" in result:
            return result
        slow = []
        for row in result["rows"]:
            sql_text, exec_count, avg_ms, max_ms, rows_examined = row
            slow.append({
                "sql": (sql_text or "")[:500],
                "exec_count": int(exec_count),
                "avg_ms": float(avg_ms),
                "max_ms": float(max_ms),
                "rows_examined": int(rows_examined),
                "is_slow": float(avg_ms) >= threshold_ms,
            })
        slow.sort(key=lambda x: x["avg_ms"], reverse=True)
        return {"slow_queries": slow, "threshold_ms": threshold_ms}

    async def connection_status(self) -> Dict[str, Any]:
        """连接池状态：当前连接数 / 上限 / 运行中线程"""
        status = {}
        for key in ["Threads_connected", "Threads_running", "Max_used_connections"]:
            r = await self.query(f"SHOW GLOBAL STATUS LIKE '{key}'")
            if "rows" in r and r["rows"]:
                status[key] = int(r["rows"][0][1])
        r = await self.query("SHOW VARIABLES LIKE 'max_connections'")
        if "rows" in r and r["rows"]:
            status["max_connections"] = int(r["rows"][0][1])
        return status

    async def health(self) -> Dict[str, Any]:
        """数据库健康状态：版本、连接、QPS、存储"""
        health: Dict[str, Any] = {}
        r = await self.query("SELECT VERSION()")
        if "error" in r:
            return r
        health["version"] = r["rows"][0][0]

        for key in ["Threads_connected", "Questions", "Uptime", "Slow_queries"]:
            r2 = await self.query(f"SHOW GLOBAL STATUS LIKE '{key}'")
            if "rows" in r2 and r2["rows"]:
                health[key] = int(r2["rows"][0][1])
        r3 = await self.query("SHOW VARIABLES LIKE 'max_connections'")
        if "rows" in r3 and r3["rows"]:
            health["max_connections"] = int(r3["rows"][0][1])

        uptime = health.get("Uptime", 1) or 1
        health["qps"] = round(health.get("Questions", 0) / uptime, 2)

        # 存储：目标库所有表的数据量
        r4 = await self.query(
            "SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) "
            "FROM information_schema.tables WHERE table_schema = %s",
            (self._cfg["database"],),
        )
        if "rows" in r4 and r4["rows"] and r4["rows"][0][0] is not None:
            health["database_size_mb"] = float(r4["rows"][0][0])
        return health

    async def execute_readonly(self, sql: str, max_rows: int = 100) -> Dict[str, Any]:
        """只读 SQL：强制 LIMIT，账号权限兜底"""
        stripped = sql.strip().rstrip(";")
        if "limit" not in stripped.lower():
            stripped = f"{stripped} LIMIT {int(max_rows)}"
        start = time.time()
        result = await self.query(stripped)
        result["execution_time_ms"] = int((time.time() - start) * 1000)
        return result


_mysql_backend: Optional[MySQLBackend] = None


def get_mysql_backend() -> MySQLBackend:
    global _mysql_backend
    if _mysql_backend is None:
        _mysql_backend = MySQLBackend()
    return _mysql_backend


# ================= Prometheus =================

class PrometheusBackend:
    """Prometheus HTTP API 真实后端"""

    def __init__(self):
        self.base_url = get_prometheus_config()["base_url"].rstrip("/")

    async def query(self, promql: str) -> Dict[str, Any]:
        """instant query：返回 {"data": ...} 或 {"error": ...}"""
        import requests

        def _run():
            resp = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=PROM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            payload = await asyncio.to_thread(_run)
            if payload.get("status") != "success":
                return {"error": f"PromQL failed: {payload.get('error', 'unknown')}"}
            return {"data": payload["data"]["result"]}
        except Exception as e:
            logger.error(f"Prometheus query failed: {type(e).__name__}: {str(e)[:200]}")
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    async def query_range(self, promql: str, duration_minutes: int = 15, step: str = "60s") -> Dict[str, Any]:
        """range query：返回时间序列"""
        import requests

        def _run():
            end = time.time()
            start = end - duration_minutes * 60
            resp = requests.get(
                f"{self.base_url}/api/v1/query_range",
                params={"query": promql, "start": start, "end": end, "step": step},
                timeout=PROM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            payload = await asyncio.to_thread(_run)
            if payload.get("status") != "success":
                return {"error": f"PromQL failed: {payload.get('error', 'unknown')}"}
            return {"data": payload["data"]["result"]}
        except Exception as e:
            logger.error(f"Prometheus range query failed: {type(e).__name__}: {str(e)[:200]}")
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    async def is_healthy(self) -> bool:
        import requests

        def _run():
            resp = requests.get(f"{self.base_url}/-/ready", timeout=3)
            return resp.status_code == 200

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return False


_prom_backend: Optional[PrometheusBackend] = None


def get_prometheus_backend() -> PrometheusBackend:
    global _prom_backend
    if _prom_backend is None:
        _prom_backend = PrometheusBackend()
    return _prom_backend


# ================= Kubernetes =================

class K8sBackend:
    """
    K8s 真实后端（kubernetes python client）。
    集群不可达时所有方法返回 {"error": ...}，工具层转降级消息。
    """

    def __init__(self):
        self._cfg = get_k8s_config()
        self._core_v1 = None
        self._load_error: Optional[str] = None
        self._init_client()

    def _init_client(self):
        try:
            from kubernetes import client, config as k8s_config
            if self._cfg["api_server"]:
                cfg = client.Configuration()
                cfg.host = self._cfg["api_server"]
                cfg.verify_ssl = False
                if self._cfg["token"]:
                    cfg.api_key = {"authorization": f"Bearer {self._cfg['token']}"}
                api_client = client.ApiClient(cfg)
            else:
                try:
                    k8s_config.load_kube_config()
                except Exception:
                    k8s_config.load_incluster_config()
                api_client = None
            self._core_v1 = client.CoreV1Api(api_client) if api_client else client.CoreV1Api()
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {str(e)[:150]}"
            logger.warning(f"K8s client init failed: {self._load_error}")

    @property
    def available(self) -> bool:
        return self._core_v1 is not None

    def _err(self) -> Dict[str, Any]:
        return {"error": f"K8s 集群不可达: {self._load_error or 'client not initialized'}"}

    async def list_pods_by_label(self, namespace: str, label_selector: str) -> Dict[str, Any]:
        def _run():
            return self._core_v1.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector, _request_timeout=QUERY_TIMEOUT_SECONDS
            )

        try:
            pods = await asyncio.to_thread(_run)
            items = []
            for p in pods.items:
                restarts = sum(cs.restart_count for cs in (p.status.container_statuses or []))
                items.append({
                    "name": p.metadata.name,
                    "status": p.status.phase,
                    "restarts": restarts,
                    "node": p.spec.node_name,
                    "ip": p.status.pod_ip,
                })
            return {"pods": items}
        except Exception as e:
            logger.error(f"K8s list_pods failed: {type(e).__name__}: {str(e)[:150]}")
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    async def get_pod_logs(self, namespace: str, pod_name: str, tail_lines: int = 100,
                           container: Optional[str] = None) -> Dict[str, Any]:
        def _run():
            kwargs = dict(namespace=namespace, name=pod_name, tail_lines=tail_lines,
                          _request_timeout=QUERY_TIMEOUT_SECONDS)
            if container:
                kwargs["container"] = container
            return self._core_v1.read_namespaced_pod_log(**kwargs)

        try:
            logs = await asyncio.to_thread(_run)
            return {"logs": logs or ""}
        except Exception as e:
            logger.error(f"K8s get_pod_logs failed: {type(e).__name__}: {str(e)[:150]}")
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    async def describe_pod(self, namespace: str, pod_name: str) -> Dict[str, Any]:
        def _run():
            pod = self._core_v1.read_namespaced_pod(name=pod_name, namespace=namespace,
                                                     _request_timeout=QUERY_TIMEOUT_SECONDS)
            return pod

        try:
            pod = await asyncio.to_thread(_run)
            cs = (pod.status.container_statuses or [None])[0]
            last_state = {}
            if cs and cs.last_state and cs.last_state.terminated:
                t = cs.last_state.terminated
                last_state = {"terminated": {"reason": t.reason, "exit_code": t.exit_code}}
            return {
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "restarts": cs.restart_count if cs else 0,
                "last_state": last_state,
                "node": pod.spec.node_name,
                "ip": pod.status.pod_ip,
            }
        except Exception as e:
            logger.error(f"K8s describe_pod failed: {type(e).__name__}: {str(e)[:150]}")
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}


_k8s_backend: Optional[K8sBackend] = None


def get_k8s_backend() -> K8sBackend:
    global _k8s_backend
    if _k8s_backend is None:
        _k8s_backend = K8sBackend()
    return _k8s_backend
