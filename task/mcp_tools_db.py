"""
数据库诊断 MCP Tools（真实后端版）
封装慢查询分析、只读 SQL、健康检查为 MCP 工具。后端：MySQL 9.x performance_schema。
"""
import asyncio
from typing import Any, Dict
import logging

from mcp_server import mcp_server
from backends import get_mysql_backend

logger = logging.getLogger("MCPToolsDB")


async def analyze_slow_queries(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """分析数据库慢查询"""
    database = arguments.get("database", "production")
    threshold_ms = arguments.get("threshold_ms", 1000)
    limit = arguments.get("limit", 10)

    mysql = get_mysql_backend()
    result = await mysql.slow_queries(threshold_ms=threshold_ms, limit=limit)
    if "error" in result:
        return {"database": database, "error": result["error"], "queries": []}

    slow = result.get("slow_queries", [])
    slow_only = [q for q in slow if q.get("is_slow")]
    return {
        "database": database,
        "threshold_ms": threshold_ms,
        "total_slow_queries": len(slow_only),
        "queries": slow,
        "summary": {
            "avg_execution_time_ms": round(sum(q["avg_ms"] for q in slow) / len(slow), 1) if slow else 0,
            "max_execution_time_ms": max((q["max_ms"] for q in slow), default=0),
            "total_rows_examined": sum(q["rows_examined"] for q in slow),
        },
    }


async def execute_readonly_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行只读 SQL 查询（禁止写操作）"""
    database = arguments.get("database", "production")
    query = arguments.get("query", "")
    max_rows = arguments.get("max_rows", 100)

    if not query:
        raise ValueError("query is required")

    # 安全检查：只允许 SELECT 语句
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for safety reasons")

    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            raise ValueError(f"Forbidden keyword detected: {keyword}")

    mysql = get_mysql_backend()
    result = await mysql.execute_readonly(query, max_rows=max_rows)
    if "error" in result:
        return {"database": database, "query": query, "error": result["error"]}

    columns = result.get("columns", [])
    rows = result.get("rows", [])
    return {
        "database": database,
        "query": query,
        "execution_time_ms": result.get("execution_time_ms", 0),
        "columns": columns,
        "row_count": len(rows),
        "rows": [dict(zip(columns, [str(v) if v is not None else None for v in row])) for row in rows[:max_rows]],
    }


async def check_database_health(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """检查数据库健康状态"""
    database = arguments.get("database", "production")

    mysql = get_mysql_backend()
    health = await mysql.health()
    if "error" in health:
        return {"database": database, "status": "unreachable", "error": health["error"]}

    threads = health.get("Threads_connected", 0)
    max_conn = health.get("max_connections", 151)
    usage_pct = round(threads / max_conn * 100, 1) if max_conn else 0

    return {
        "database": database,
        "status": "healthy",
        "version": health.get("version"),
        "connections": {
            "active": threads,
            "max": max_conn,
            "usage_percent": usage_pct,
        },
        "performance": {
            "queries_per_second": health.get("qps", 0),
            "uptime_seconds": health.get("Uptime", 0),
            "slow_queries_total": health.get("Slow_queries", 0),
        },
        "storage": {
            "database_size_mb": health.get("database_size_mb", 0),
        },
    }


async def analyze_lock_contention(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """分析数据库锁竞争情况（MySQL 9.x performance_schema.data_locks）"""
    database = arguments.get("database", "production")

    mysql = get_mysql_backend()
    # MySQL 8+/9 的锁视图
    result = await mysql.query(
        "SELECT ENGINE_TRANSACTION_ID, OBJECT_NAME, LOCK_TYPE, LOCK_MODE, LOCK_STATUS "
        "FROM performance_schema.data_locks LIMIT 20"
    )
    if "error" in result:
        return {"database": database, "total_locks": 0, "locks": [], "severity": "unknown",
                "note": result["error"]}

    locks = []
    for row in result.get("rows", []):
        locks.append({
            "trx_id": str(row[0]) if row[0] else None,
            "table": row[1],
            "lock_type": row[2],
            "lock_mode": row[3],
            "lock_status": row[4],
        })

    waiting = [l for l in locks if l.get("lock_status") == "WAITING"]
    return {
        "database": database,
        "total_locks": len(locks),
        "waiting_locks": len(waiting),
        "locks": locks[:10],
        "severity": "none" if not locks else ("low" if len(waiting) == 0 else "high"),
    }


# 注册数据库相关工具到 MCP Server
def register_db_tools():
    """注册所有数据库工具"""

    mcp_server.register_tool(
        name="analyze_slow_queries",
        description="分析数据库慢查询日志，找出执行时间最长的查询",
        input_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "default": "production"},
                "threshold_ms": {
                    "type": "integer",
                    "description": "慢查询阈值（毫秒）",
                    "default": 1000
                },
                "limit": {"type": "integer", "default": 10}
            }
        },
        handler=analyze_slow_queries,
        timeout_seconds=15
    )

    mcp_server.register_tool(
        name="execute_readonly_query",
        description="执行只读 SQL 查询（仅允许 SELECT，禁止写操作）",
        input_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "default": "production"},
                "query": {"type": "string", "description": "SQL 查询语句（仅 SELECT）"},
                "max_rows": {"type": "integer", "default": 100}
            },
            "required": ["query"]
        },
        handler=execute_readonly_query,
        timeout_seconds=30
    )

    mcp_server.register_tool(
        name="check_database_health",
        description="检查数据库健康状态（连接、性能、存储）",
        input_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "default": "production"}
            }
        },
        handler=check_database_health,
        timeout_seconds=10
    )

    mcp_server.register_tool(
        name="analyze_lock_contention",
        description="分析数据库锁竞争情况",
        input_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "default": "production"}
            }
        },
        handler=analyze_lock_contention,
        timeout_seconds=15
    )

    logger.info("Registered 4 database tools to MCP Server")


# 自动注册工具
register_db_tools()
