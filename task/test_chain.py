"""
链路联调测试：单 Agent 主链路 + 多 Agent 协作链路（真实 LLM / Redis / Milvus）
用法：cd D:\AIOps\task && python test_chain.py
"""
import asyncio
import sys

import settings  # noqa: F401  触发 .env 加载

from main import create_llm_instance, run_single_agent, run_multi_agent
from route import build_graph


def show(title, result):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"状态: {result.get('status')}")
    if result.get("error"):
        print(f"错误: {result['error']}")
    print(f"会话: {result.get('session_id')}")
    print(f"耗时: {result.get('duration_seconds', 0):.1f}s")
    if result.get("execution_history") is not None:
        print(f"执行步骤数: {len(result['execution_history'])}")
        for i, h in enumerate(result["execution_history"][:6]):
            step = h.get("step_info", {})
            print(f"  [{i+1}] {step.get('action')} on {step.get('target')} -> {str(h.get('result_summary',''))[:70]}")
    if result.get("specialist_findings") is not None:
        print(f"专家发现: {list(result['specialist_findings'].keys())}, handoff={result.get('handoff_count')}")
        for role, finding in result["specialist_findings"].items():
            print(f"  [{role}] {str(finding)[:90]}...")
    answer = result.get("final_answer", "") or ""
    print(f"最终答案(前300字): {answer[:300]}")
    return result.get("status") == "SUCCESS"


async def main():
    ok_all = True

    # 0. 确认 checkpointer 类型（Redis 持久化 or 内存降级）
    try:
        graph = await build_graph()
        cp_type = type(graph.checkpointer).__name__
        print(f"[Checkpointer] {cp_type}")
        if "Redis" not in cp_type:
            print("  ⚠️ Redis 检查点未生效（降级内存）")
    except Exception as e:
        print(f"[Checkpointer] 检查失败: {e}")

    llm = create_llm_instance()

    # 1. 单 Agent：故障排查链路（router→topology→planner→executor→工具→反思→critic→RCA→分发）
    #    使用数据库类故障（本机 MySQL 真实可用），展示完整工具链成功路径
    single_query = "订单服务接口响应变慢，疑似数据库慢查询或连接池不足，请排查数据库侧根因"
    try:
        single_result = await asyncio.wait_for(
            run_single_agent(single_query, severity="P2", llm=llm), timeout=600
        )
    except asyncio.TimeoutError:
        single_result = {"status": "FAILED", "error": "TIMEOUT(600s)"}
    ok_all &= show("[单 Agent 排查链路]", single_result)

    # 2. 多 Agent：Supervisor→专家子图→Handoff/仲裁
    multi_query = "订单服务接口延迟飙升，疑似下游依赖或数据库问题，请多专家协作排查"
    try:
        multi_result = await asyncio.wait_for(
            run_multi_agent(multi_query, severity="P2", llm=llm), timeout=600
        )
    except asyncio.TimeoutError:
        multi_result = {"status": "FAILED", "error": "TIMEOUT(600s)"}
    ok_all &= show("[多 Agent 协作链路]", multi_result)

    print("\n" + "=" * 60)
    print("ALL CHAIN TESTS PASSED" if ok_all else "SOME CHAIN TESTS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    asyncio.run(main())
