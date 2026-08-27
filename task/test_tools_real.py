"""
独立测试脚本：逐个检验真实后端接入后的每个工具
用法：cd D:\\AIOps\\task && python test_tools_real.py

逐个调用每个类工具，断言返回结构合法（有内容/有降级信息），输出 PASS/FAIL 汇总。
依赖：MySQL(3306, root) / Prometheus(9090) / K8s(可选，无集群时验证降级路径)。
"""
import asyncio
import json
import sys

import settings  # noqa: F401  触发 .env 加载

PASS, FAIL = [], []


def report(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    (PASS if ok else FAIL).append(name)


def _parse(text: str):
    """尝试解析 JSON，失败返回 None"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ================= tool.py 的 6 个工具 =================

async def test_tool_py():
    from tool import (
        get_pod_status, fetch_k8s_logs, describe_pod, restart_service,
        query_slow_sql, check_db_connections,
    )

    # 1. get_pod_status
    try:
        out = await get_pod_status.ainvoke({"service_name": "payment-service"})
        data = _parse(out)
        ok = data is not None and "pods" in data and "summary" in data
        report("get_pod_status", ok, data.get("summary", "")[:60] if data else "解析失败")
    except Exception as e:
        report("get_pod_status", False, f"{type(e).__name__}: {e}")

    # 2. fetch_k8s_logs（K8s 不可达时应返回降级提示而非崩溃）
    try:
        out = await fetch_k8s_logs.ainvoke({"pod_name": "payment-service-abc123", "lines": 20})
        ok = isinstance(out, str) and len(out) > 0
        report("fetch_k8s_logs", ok, out[:60].replace("\n", " "))
    except Exception as e:
        report("fetch_k8s_logs", False, f"{type(e).__name__}: {e}")

    # 3. describe_pod
    try:
        out = await describe_pod.ainvoke({"pod_name": "payment-service-abc123"})
        data = _parse(out)
        ok = data is not None and ("name" in data or "error" in data)
        report("describe_pod", ok, "有结构" if data else "解析失败")
    except Exception as e:
        report("describe_pod", False, f"{type(e).__name__}: {e}")

    # 4. restart_service（危险操作，验证结构不真实执行）
    try:
        out = await restart_service.ainvoke({"service_name": "payment-service"})
        data = _parse(out)
        ok = data is not None and "status" in data
        report("restart_service", ok, data.get("message", "")[:50] if data else "解析失败")
    except Exception as e:
        report("restart_service", False, f"{type(e).__name__}: {e}")

    # 5. query_slow_sql（MySQL 真实后端）
    try:
        out = await query_slow_sql.ainvoke({"service_name": "payment-service", "threshold_ms": 10})
        data = _parse(out)
        ok = data is not None and "slow_queries" in data and "summary" in data
        report("query_slow_sql", ok, data.get("summary", "")[:60] if data else "解析失败")
    except Exception as e:
        report("query_slow_sql", False, f"{type(e).__name__}: {e}")

    # 6. check_db_connections（MySQL 真实后端）
    try:
        out = await check_db_connections.ainvoke({"service_name": "payment-service"})
        data = _parse(out)
        ok = data is not None and "active_connections" in data and "status" in data
        report("check_db_connections", ok,
               f"active={data.get('active_connections')}/{data.get('max_connections')} {data.get('status')}" if data else "解析失败")
    except Exception as e:
        report("check_db_connections", False, f"{type(e).__name__}: {e}")


# ================= mcp_tools_monitor.py 的 4 个工具 =================

async def test_monitor_tools():
    from mcp_tools_monitor import (
        query_prometheus_metrics, fetch_k8s_pod_logs, get_k8s_pod_status, query_service_dependencies,
    )

    # 7. query_prometheus_metrics
    try:
        out = await query_prometheus_metrics({"service": "payment-service", "metric_type": "all", "duration_minutes": 5})
        ok = isinstance(out, dict) and ("status" in out)
        report("query_prometheus_metrics", ok, f"status={out.get('status')}")
    except Exception as e:
        report("query_prometheus_metrics", False, f"{type(e).__name__}: {e}")

    # 8. fetch_k8s_pod_logs
    try:
        out = await fetch_k8s_pod_logs({"pod_name": "payment-service-abc123", "lines": 10})
        ok = isinstance(out, dict) and ("status" in out or "error" in out or "logs" in out)
        report("fetch_k8s_pod_logs", ok, str(out.get("status") or out.get("error") or "has logs")[:50])
    except Exception as e:
        report("fetch_k8s_pod_logs", False, f"{type(e).__name__}: {e}")

    # 9. get_k8s_pod_status
    try:
        out = await get_k8s_pod_status({"pod_name": "payment-service-abc123"})
        ok = isinstance(out, dict) and ("status" in out or "error" in out)
        report("get_k8s_pod_status", ok, str(out.get("status"))[:40])
    except Exception as e:
        report("get_k8s_pod_status", False, f"{type(e).__name__}: {e}")

    # 10. query_service_dependencies（走真实拓扑图）
    try:
        out = await query_service_dependencies({"service": "Order-Service"})
        ok = isinstance(out, dict) and "dependencies" in out
        deps = out.get("dependencies", {})
        report("query_service_dependencies", ok,
               f"downstream={deps.get('downstream')}")
    except Exception as e:
        report("query_service_dependencies", False, f"{type(e).__name__}: {e}")


# ================= mcp_tools_db.py 的 4 个工具 =================

async def test_db_tools():
    from mcp_tools_db import (
        analyze_slow_queries, execute_readonly_query, check_database_health, analyze_lock_contention,
    )

    # 11. analyze_slow_queries
    try:
        out = await analyze_slow_queries({"database": "aiops", "threshold_ms": 1, "limit": 5})
        ok = isinstance(out, dict) and "queries" in out and "summary" in out
        report("analyze_slow_queries", ok,
               f"slow={out.get('total_slow_queries')}, max_ms={out.get('summary', {}).get('max_execution_time_ms')}")
    except Exception as e:
        report("analyze_slow_queries", False, f"{type(e).__name__}: {e}")

    # 12. execute_readonly_query（合法 + 攻击用例）
    try:
        ok_result = await execute_readonly_query({"database": "aiops", "query": "SELECT COUNT(*) AS cnt FROM orders", "max_rows": 5})
        ok = isinstance(ok_result, dict) and ("rows" in ok_result or "error" in ok_result)

        # 攻击语句应被拒绝
        rejected = 0
        for evil in ["DROP TABLE orders", "SELECT 1; DELETE FROM orders", "UPDATE orders SET amount=0"]:
            try:
                await execute_readonly_query({"database": "aiops", "query": evil})
            except ValueError:
                rejected += 1
        ok = ok and rejected == 3
        report("execute_readonly_query", ok, f"合法查询通过, {rejected}/3 攻击语句拦截")
    except Exception as e:
        report("execute_readonly_query", False, f"{type(e).__name__}: {e}")

    # 13. check_database_health
    try:
        out = await check_database_health({"database": "aiops"})
        ok = isinstance(out, dict) and "status" in out and "connections" in out
        report("check_database_health", ok,
               f"status={out.get('status')}, qps={out.get('performance', {}).get('queries_per_second')}")
    except Exception as e:
        report("check_database_health", False, f"{type(e).__name__}: {e}")

    # 14. analyze_lock_contention
    try:
        out = await analyze_lock_contention({"database": "aiops"})
        ok = isinstance(out, dict) and "total_locks" in out and "severity" in out
        report("analyze_lock_contention", ok,
               f"locks={out.get('total_locks')}, severity={out.get('severity')}")
    except Exception as e:
        report("analyze_lock_contention", False, f"{type(e).__name__}: {e}")


async def main():
    print("=" * 60)
    print("AIOps Agent 真实工具接入测试")
    print("=" * 60)

    print("\n--- tool.py (LangChain Tools) ---")
    await test_tool_py()

    print("\n--- mcp_tools_monitor.py ---")
    await test_monitor_tools()

    print("\n--- mcp_tools_db.py ---")
    await test_db_tools()

    print("\n" + "=" * 60)
    total = len(PASS) + len(FAIL)
    print(f"结果: {len(PASS)}/{total} 通过")
    if FAIL:
        print(f"未通过: {FAIL}")
        sys.exit(1)
    print("ALL TOOL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
