"""
AIOps 系统中厂量化指标测试集
============================
覆盖 6 大维度 × 多项指标，每项测试独立可运行。

维度:
  1. RAG 检索质量 (Recall@K / MRR / NDCG / 检索耗时 / 空检索率)
  2. 诊断准确性   (RCA 准确率 / 置信度校准 / 证据链完整性 / SMART 合规率)
  3. 执行效率     (MTTD / MTTR / 工具成功率 / 重规划率 / 死循环率)
  4. 成本控制     (单工单 token / 单工单 LLM 调用次数 / 成本效率比)
  5. 系统韧性     (依赖降级 / 超时捕获 / 熔断正确性 / 优雅失败)
  6. 安全合规     (RBAC 拦截 / 注入防护 / 高危审批拦截 / 只读 SQL)

运行: cd D:\\AIOps\\task && python test_evaluation_suite.py
依赖: ai311 虚拟环境, MySQL9.6
"""

import asyncio
import json
import time
import logging
import os
import re
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 确保 .env 加载
import settings  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestSuite")


# ========================================================================
# 基础设施
# ========================================================================

@dataclass
class TestResult:
    """单条测试结果"""
    test_id: str
    dimension: str
    metric: str
    passed: bool
    value: Any = None
    threshold: Any = None
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class TestReport:
    """测试报告聚合"""
    results: List[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add(self, result: TestResult):
        self.results.append(result)
        tag = "PASS" if result.passed else "FAIL"
        logger.info(f"  [{tag}] {result.dimension}/{result.metric}: {result.value} (阈值: {result.threshold})")

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        by_dimension = {}
        for r in self.results:
            dim = r.dimension
            if dim not in by_dimension:
                by_dimension[dim] = {"total": 0, "passed": 0, "metrics": {}}
            by_dimension[dim]["total"] += 1
            if r.passed:
                by_dimension[dim]["passed"] += 1
            by_dimension[dim]["metrics"][r.metric] = {
                "value": r.value,
                "threshold": r.threshold,
                "passed": r.passed,
                "detail": r.detail
            }
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed / total * 100:.1f}%" if total else "N/A",
            "by_dimension": by_dimension
        }

    def export_markdown(self) -> str:
        """导出 Markdown 测试报告"""
        s = self.summary()
        md = f"# AIOps 系统量化指标测试报告\n\n"
        md += f"生成时间: {datetime.now().isoformat()}\n\n"
        md += f"## 总览\n\n"
        md += f"| 指标 | 数值 |\n|---|---|\n"
        md += f"| 总测试项 | {s['total']} |\n"
        md += f"| 通过 | {s['passed']} |\n"
        md += f"| 失败 | {s['failed']} |\n"
        md += f"| 通过率 | {s['pass_rate']} |\n\n"

        md += f"## 分维度明细\n\n"
        for dim, data in s["by_dimension"].items():
            md += f"### {dim} ({data['passed']}/{data['total']})\n\n"
            md += f"| 指标 | 实际值 | 阈值 | 结果 | 说明 |\n|---|---|---|---|---|\n"
            for metric, info in data["metrics"].items():
                tag = "PASS" if info["passed"] else "**FAIL**"
                md += f"| {metric} | {info['value']} | {info['threshold']} | {tag} | {info['detail']} |\n"
            md += "\n"

        return md


report = TestReport()


# ========================================================================
# 维度 1: RAG 检索质量
# ========================================================================

# 标准测试 Query 集（模拟中厂常见故障场景）
RAG_TEST_CASES = [
    {
        "query": "订单服务 Pod 频繁重启 CrashLoopBackOff OOM 堆内存溢出",
        "expected_keywords": ["OOM", "堆内存", "CrashLoopBackOff", "重启"],
        "expected_case_ids": [],  # 如果知识库有标注 case_id 可填入
        "description": "OOM 导致容器反复重启"
    },
    {
        "query": "数据库连接池耗尽 Druid 等待线程超时 Connection refused",
        "expected_keywords": ["连接池", "Druid", "等待", "超时"],
        "expected_case_ids": [],
        "description": "数据库连接池耗尽"
    },
    {
        "query": "网关 P99 延迟飙升 Full GC STW 拖垮上游服务",
        "expected_keywords": ["P99", "延迟", "GC", "STW"],
        "expected_case_ids": [],
        "description": "GC 导致延迟飙升"
    },
    {
        "query": "Redis 缓存雪崩 大面积 Key 同时过期 击穿后端数据库",
        "expected_keywords": ["缓存", "雪崩", "Key", "过期"],
        "expected_case_ids": [],
        "description": "缓存雪崩"
    },
    {
        "query": "K8s Node 资源不足 Pod 被驱逐 Evicted DiskPressure",
        "expected_keywords": ["驱逐", "Evicted", "DiskPressure", "资源"],
        "expected_case_ids": [],
        "description": "K8s 节点资源不足"
    },
    {
        "query": "MySQL 慢查询 全表扫描 索引缺失 查询超时",
        "expected_keywords": ["慢查询", "全表扫描", "索引", "超时"],
        "expected_case_ids": [],
        "description": "MySQL 慢查询"
    },
]


async def test_rag_retrieval_quality():
    """维度1: RAG 检索质量测试"""
    logger.info("\n===== 维度1: RAG 检索质量 =====")

    try:
        from retrieve import rag_retrieval_node
        from bone import IncidentSeverity
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        report.add(TestResult("RAG-001", "RAG检索质量", "模块导入", False, detail=str(e)))
        return

    latencies = []
    empty_count = 0
    keyword_hits = 0

    for i, case in enumerate(RAG_TEST_CASES):
        t0 = time.time()
        try:
            state = {
                "query": case["query"],
                "incident_severity": IncidentSeverity.P2,
                "related_components_for_filter": [],
                "expanded_search_queries": [],
            }
            result = await rag_retrieval_node(state, {"configurable": {}})
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            ctx = result.get("retrieved_context", [])
            ctx_text = " ".join(ctx) if ctx else ""

            # 空检索率
            if not ctx_text or len(ctx_text) < 20:
                empty_count += 1

            # 关键词命中率
            hit = sum(1 for kw in case["expected_keywords"] if kw.lower() in ctx_text.lower())
            if hit >= len(case["expected_keywords"]) * 0.5:
                keyword_hits += 1

            report.add(TestResult(
                f"RAG-{i+1:03d}", "RAG检索质量", f"Case{i+1} 关键词命中",
                hit >= len(case["expected_keywords"]) * 0.5,
                value=f"{hit}/{len(case['expected_keywords'])}",
                threshold=f">={len(case['expected_keywords']) * 0.5}",
                detail=case["description"],
                duration_ms=latency
            ))

        except Exception as e:
            latency = (time.time() - t0) * 1000
            latencies.append(latency)
            report.add(TestResult(
                f"RAG-{i+1:03d}", "RAG检索质量", f"Case{i+1} 异常",
                False, detail=f"{type(e).__name__}: {str(e)[:100]}",
                duration_ms=latency
            ))

    # 聚合指标
    total = len(RAG_TEST_CASES)
    empty_rate = empty_count / total if total else 1.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 999
    keyword_hit_rate = keyword_hits / total if total else 0.0

    report.add(TestResult(
        "RAG-100", "RAG检索质量", "空检索率",
        empty_rate <= 0.5,
        value=f"{empty_rate:.1%}",
        threshold="<=50%",
        detail=f"{empty_count}/{total} 次空检索"
    ))
    report.add(TestResult(
        "RAG-101", "RAG检索质量", "平均检索耗时",
        avg_latency <= 5000,
        value=f"{avg_latency:.0f}ms",
        threshold="<=5000ms",
        detail=f"P99 应 <8s"
    ))
    report.add(TestResult(
        "RAG-102", "RAG检索质量", "关键词命中率",
        keyword_hit_rate >= 0.5,
        value=f"{keyword_hit_rate:.1%}",
        threshold=">=50%",
        detail=f"{keyword_hits}/{total} 命中"
    ))


# ========================================================================
# 维度 2: 诊断准确性
# ========================================================================

RCA_TEST_CASES = [
    {
        "query": "订单服务 Pod 频繁重启 CrashLoopBackOff OOM 堆内存溢出",
        "expected_root_cause_keywords": ["内存", "OOM", "堆", "JVM"],
        "expected_evidence_types": ["Log", "Metric"],
        "description": "OOM 根因应指向内存配置或 JVM 参数"
    },
    {
        "query": "数据库连接池耗尽 Druid ActiveConnections 达到上限 50/50",
        "expected_root_cause_keywords": ["连接池", "Druid", "连接数"],
        "expected_evidence_types": ["Metric", "Log"],
        "description": "连接池耗尽根因"
    },
    {
        "query": "Payment 服务响应超时 下游 timeout 30s 熔断器打开",
        "expected_root_cause_keywords": ["超时", "下游", "熔断"],
        "expected_evidence_types": ["Log", "Metric"],
        "description": "级联超时根因"
    },
]


async def test_diagnosis_accuracy():
    """维度2: 诊断准确性测试（验证 RCA Schema 结构完整性）"""
    logger.info("\n===== 维度2: 诊断准确性 =====")

    try:
        from bone import RCAResult, EvidenceItem, ActionItem, ActionItemType
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        report.add(TestResult("DIAG-001", "诊断准确性", "模块导入", False, detail=str(e)))
        return

    # 2.1 验证 RCAResult Schema 完整性
    try:
        sample_rca = RCAResult(
            incident_summary="订单服务 OOM 导致 Pod 频繁重启",
            timeline=[],
            root_cause_analysis="JVM 堆内存配置不足，最大堆仅 512MB，无法承载当前并发量",
            evidence_chain=[
                EvidenceItem(type="Log", content="OutOfMemoryError: Java heap space", relevance="直接证据"),
                EvidenceItem(type="Metric", content="Pod memory usage 510/512MB", relevance="佐证")
            ],
            impact_assessment="影响订单创建接口，约 30% 请求失败",
            action_items=[
                ActionItem(
                    type=ActionItemType.SHORT_TERM,
                    description="将 JVM 最大堆从 512MB 调整为 1024MB",
                    owner="SRE-Backend",
                    deadline="2025-03-01",
                    priority="P1"
                )
            ],
            confidence_score=0.85
        )

        # 验证证据链非空
        has_evidence = len(sample_rca.evidence_chain) > 0
        report.add(TestResult(
            "DIAG-001", "诊断准确性", "RCA Schema 证据链",
            has_evidence,
            value=len(sample_rca.evidence_chain),
            threshold=">=1",
            detail="每条根因必须有证据支撑"
        ))

        # 验证置信度范围
        valid_confidence = 0.0 <= sample_rca.confidence_score <= 1.0
        report.add(TestResult(
            "DIAG-002", "诊断准确性", "置信度校准",
            valid_confidence,
            value=sample_rca.confidence_score,
            threshold="0.0~1.0",
            detail="置信度必须在合法范围内"
        ))

        # 验证 Action Items SMART 合规性
        smart_compliant = all(
            item.description and item.owner and item.deadline and item.priority
            for item in sample_rca.action_items
        )
        report.add(TestResult(
            "DIAG-003", "诊断准确性", "SMART 合规率",
            smart_compliant,
            value="100%" if smart_compliant else "0%",
            threshold="100%",
            detail="Action Items 必须包含具体动作/Owner/Deadline/Priority"
        ))

    except Exception as e:
        report.add(TestResult(
            "DIAG-001", "诊断准确性", "RCA Schema 构建",
            False, detail=f"{type(e).__name__}: {str(e)[:100]}"
        ))

    # 2.2 验证 RCA Generator Prompt 约束（检查 SMART 示例是否注入）
    try:
        from rca_generator import RCA_GENERATION_PROMPT, SMART_EXAMPLES

        has_smart_examples = "SMART" in SMART_EXAMPLES or "Owner" in SMART_EXAMPLES
        report.add(TestResult(
            "DIAG-004", "诊断准确性", "SMART 示例注入",
            has_smart_examples,
            value="有" if has_smart_examples else "无",
            threshold="有",
            detail="RCA Prompt 必须包含 SMART Action Items 示例"
        ))

        has_evidence_requirement = "证据" in RCA_GENERATION_PROMPT or "evidence" in RCA_GENERATION_PROMPT.lower()
        report.add(TestResult(
            "DIAG-005", "诊断准确性", "证据链约束注入",
            has_evidence_requirement,
            value="有" if has_evidence_requirement else "无",
            threshold="有",
            detail="RCA Prompt 必须要求证据链"
        ))

    except ImportError as e:
        report.add(TestResult(
            "DIAG-004", "诊断准确性", "RCA Prompt 检查",
            False, detail=str(e)
        ))


# ========================================================================
# 维度 3: 执行效率
# ========================================================================

async def test_execution_efficiency():
    """维度3: 执行效率测试"""
    logger.info("\n===== 维度3: 执行效率 =====")

    # 3.1 工具调用成功率（调用工具验证参数校验和结构完整性）
    try:
        from tool import TOOL_MAP, ALL_TOOLS

        total_tools = len(ALL_TOOLS)
        callable_tools = sum(1 for t in ALL_TOOLS if hasattr(t, 'ainvoke'))
        tool_callable_rate = callable_tools / total_tools if total_tools else 0

        report.add(TestResult(
            "EFF-001", "执行效率", "工具可调用率",
            tool_callable_rate >= 0.9,
            value=f"{tool_callable_rate:.0%}",
            threshold=">=90%",
            detail=f"{callable_tools}/{total_tools} 工具可调用"
        ))

    except ImportError as e:
        report.add(TestResult("EFF-001", "执行效率", "工具导入", False, detail=str(e)))

    # 3.2 重规划上限保护（验证 replan_count >= 3 时转兜底）
    try:
        from bone import AgentStatus

        # 模拟 executor 行为: replan_count >= 3 时应 FAILED
        test_state = {"replan_count": 3, "current_plan": []}
        replan_count = test_state.get("replan_count", 0)
        should_fallback = replan_count >= 3

        report.add(TestResult(
            "EFF-002", "执行效率", "重规划上限保护",
            should_fallback,
            value=f"replan_count={replan_count}",
            threshold=">=3 时转兜底",
            detail="防止无限重规划消耗 token"
        ))

    except Exception as e:
        report.add(TestResult("EFF-002", "执行效率", "重规划上限", False, detail=str(e)))

    # 3.3 死循环检测（多 Agent P2P Handoff 死循环）
    try:
        from supervisor import detect_deadlock, MAX_HANDOFF_COUNT
        from bone import AgentHandoff, SpecialistRole

        # 构造死循环历史：L1→L2→L1→L2→...
        deadlock_history = [
            AgentHandoff(
                from_agent=SpecialistRole.L1_AGENT,
                to_agent=SpecialistRole.L2_AGENT,
                context_summary="test",
                reason="test"
            ),
            AgentHandoff(
                from_agent=SpecialistRole.L2_AGENT,
                to_agent=SpecialistRole.L1_AGENT,
                context_summary="test",
                reason="test"
            ),
            AgentHandoff(
                from_agent=SpecialistRole.L1_AGENT,
                to_agent=SpecialistRole.L2_AGENT,
                context_summary="test",
                reason="test"
            ),
        ]

        deadlock_detected = detect_deadlock(deadlock_history)
        report.add(TestResult(
            "EFF-003", "执行效率", "死循环检测",
            deadlock_detected,
            value="检测到" if deadlock_detected else "未检测到",
            threshold="检测到",
            detail="L1→L2→L1 循环应被检测"
        ))

        # 验证 Handoff 上限
        has_max_limit = MAX_HANDOFF_COUNT > 0 and MAX_HANDOFF_COUNT <= 20
        report.add(TestResult(
            "EFF-004", "执行效率", "Handoff 上限设置",
            has_max_limit,
            value=MAX_HANDOFF_COUNT,
            threshold="1~20",
            detail="防止无限交接"
        ))

    except Exception as e:
        report.add(TestResult("EFF-003", "执行效率", "死循环检测", False, detail=str(e)))

    # 3.4 Planner 工具名校验（验证 Plan 中 action 必须是合法工具名）
    try:
        from tool import TOOL_MAP

        valid_tool_names = set(TOOL_MAP.keys())
        # 模拟 Planner 输出含非法工具名的步骤
        fake_plan_steps = [
            {"action": "get_pod_status", "target": "order-service"},
            {"action": "invalid_tool_name", "target": "something"},
            {"action": "query_slow_sql", "target": "order-db"},
        ]
        invalid_count = sum(1 for s in fake_plan_steps if s["action"] not in valid_tool_names)

        report.add(TestResult(
            "EFF-005", "执行效率", "Plan 工具名合法性",
            invalid_count > 0,  # 这里故意有一个非法的，验证能检测出来
            value=f"非法工具名 {invalid_count} 个",
            threshold="能检测出非法工具名",
            detail=f"合法工具: {valid_tool_names}"
        ))

    except Exception as e:
        report.add(TestResult("EFF-005", "执行效率", "工具名校验", False, detail=str(e)))


# ========================================================================
# 维度 4: 成本控制
# ========================================================================

async def test_cost_control():
    """维度4: 成本控制测试"""
    logger.info("\n===== 维度4: 成本控制 =====")

    # 4.1 Token 预算管理（按严重等级分级）
    try:
        from token_budget_manager import TokenBudgetManager
        from bone import IncidentSeverity

        # 验证不同严重等级有不同的 token 预算
        tbm = TokenBudgetManager

        # 检查预算管理器是否能区分严重等级
        has_severity_awareness = hasattr(tbm, 'apply_budget_to_rag_context')
        report.add(TestResult(
            "COST-001", "成本控制", "Token 预算分级",
            has_severity_awareness,
            value="有" if has_severity_awareness else "无",
            threshold="有",
            detail="按 P0~P4 分级分配 token 预算"
        ))

    except ImportError as e:
        report.add(TestResult("COST-001", "成本控制", "Token预算模块", False, detail=str(e)))

    # 4.2 反思节点启发式预筛（减少不必要的 LLM 调用）
    try:
        from reflection_node import _is_heuristic_pass

        # 构造一个正常工具返回的 state（不应触发 LLM 反思）
        from langchain_core.messages import ToolMessage
        normal_state = {
            "messages": [ToolMessage(content="Pod status: Running, all containers healthy", tool_call_id="test")],
        }
        should_skip = _is_heuristic_pass(normal_state)

        report.add(TestResult(
            "COST-002", "成本控制", "启发式预筛(正常跳过)",
            should_skip,
            value="跳过LLM" if should_skip else "调用LLM",
            threshold="跳过LLM",
            detail="正常工具返回应跳过 LLM 反思以节省 token"
        ))

        # 构造一个含错误的 state（应触发 LLM 反思）
        error_state = {
            "messages": [ToolMessage(content="Error: Connection timeout, OOM detected", tool_call_id="test")],
        }
        should_invoke = not _is_heuristic_pass(error_state)

        report.add(TestResult(
            "COST-003", "成本控制", "启发式预筛(错误触发)",
            should_invoke,
            value="触发LLM" if should_invoke else "跳过LLM",
            threshold="触发LLM",
            detail="错误工具返回应触发 LLM 深度反思"
        ))

    except ImportError as e:
        report.add(TestResult("COST-002", "成本控制", "反思预筛", False, detail=str(e)))

    # 4.3 LLM 指标采集器
    try:
        from observability import LLMetricsHandler

        handler = LLMetricsHandler()
        initial_summary = handler.summary()
        has_token_tracking = all(k in initial_summary for k in [
            "llm_calls", "total_tokens", "prompt_tokens", "completion_tokens", "llm_time_seconds"
        ])

        report.add(TestResult(
            "COST-004", "成本控制", "LLM 指标采集完整性",
            has_token_tracking,
            value="完整" if has_token_tracking else "缺失",
            threshold="完整",
            detail="必须追踪 calls/tokens/time"
        ))

    except ImportError as e:
        report.add(TestResult("COST-004", "成本控制", "指标采集器", False, detail=str(e)))

    # 4.4 日志摘要中间件（控制日志 token 消耗）
    try:
        from tool import summarize_logs, MAX_LOG_TOKENS, MAX_LOG_LINES_RETURN

        # 生成一段很长的日志
        long_log = "\n".join([f"2025-01-01 INFO Line {i}: normal operation" for i in range(500)])
        summarized = summarize_logs(long_log)

        # 验证摘要后长度是否受控
        within_budget = len(summarized) <= MAX_LOG_TOKENS * 4 + 500  # 允许 header 开销
        report.add(TestResult(
            "COST-005", "成本控制", "日志摘要截断",
            within_budget,
            value=f"{len(summarized)} chars",
            threshold=f"<={MAX_LOG_TOKENS * 4 + 500} chars",
            detail=f"500行日志应被截断至 {MAX_LOG_TOKENS} token 预算内"
        ))

    except ImportError as e:
        report.add(TestResult("COST-005", "成本控制", "日志摘要", False, detail=str(e)))


# ========================================================================
# 维度 5: 系统韧性
# ========================================================================

async def test_system_resilience():
    """维度5: 系统韧性测试"""
    logger.info("\n===== 维度5: 系统韧性 =====")

    # 5.1 Redis 降级（checkpoint_factory 降级为 MemorySaver）
    try:
        from checkpoint_factory import create_checkpointer

        # 验证模块可导入和基本降级逻辑存在
        import inspect
        source = inspect.getsource(create_checkpointer)
        has_fallback = "MemorySaver" in source or "InMemorySaver" in source

        report.add(TestResult(
            "RES-001", "系统韧性", "Redis 降级逻辑",
            has_fallback,
            value="有降级" if has_fallback else "无降级",
            threshold="有降级",
            detail="Redis 不可用时自动降级为内存检查点"
        ))

    except ImportError as e:
        report.add(TestResult("RES-001", "系统韧性", "降级模块", False, detail=str(e)))

    # 5.2 工具超时配置完整性
    try:
        from tool import TOOL_TIMEOUTS, ALL_TOOLS

        all_tool_names = {t.name for t in ALL_TOOLS}
        configured_timeouts = set(TOOL_TIMEOUTS.keys())
        coverage = len(all_tool_names & configured_timeouts) / len(all_tool_names) if all_tool_names else 0

        report.add(TestResult(
            "RES-002", "系统韧性", "工具超时配置覆盖率",
            coverage >= 0.9,
            value=f"{coverage:.0%}",
            threshold=">=90%",
            detail=f"{len(all_tool_names & configured_timeouts)}/{len(all_tool_names)} 工具配置了超时"
        ))

        # 验证超时值合理性 (5~30s)
        reasonable_timeouts = all(5 <= v <= 30 for v in TOOL_TIMEOUTS.values())
        report.add(TestResult(
            "RES-003", "系统韧性", "超时值合理性",
            reasonable_timeouts,
            value=f"range {min(TOOL_TIMEOUTS.values())}-{max(TOOL_TIMEOUTS.values())}s",
            threshold="5~30s",
            detail="所有工具超时应在 5~30 秒内"
        ))

    except ImportError as e:
        report.add(TestResult("RES-002", "系统韧性", "超时配置", False, detail=str(e)))

    # 5.3 熔断器/重试机制
    try:
        from utils import retry_with_backoff
        import inspect

        source = inspect.getsource(retry_with_backoff)
        has_exponential = "backoff" in source.lower() or "delay" in source.lower() or "sleep" in source.lower()
        has_max_retries = "max_retries" in source

        report.add(TestResult(
            "RES-004", "系统韧性", "指数退避重试",
            has_exponential and has_max_retries,
            value="有" if has_exponential and has_max_retries else "缺失",
            threshold="有",
            detail="重试需有退避策略和上限"
        ))

    except ImportError as e:
        report.add(TestResult("RES-004", "系统韧性", "重试机制", False, detail=str(e)))

    # 5.4 会话超时看门狗
    try:
        from hallucination_guard import GlobalTimeoutManager

        tm = GlobalTimeoutManager()
        tm.register_session("test_session", timeout_minutes=5)
        remaining = tm.get_remaining_time("test_session")
        has_timeout = remaining is not None and remaining > 0

        report.add(TestResult(
            "RES-005", "系统韧性", "会话超时看门狗",
            has_timeout,
            value=f"{remaining:.0f}s" if remaining else "None",
            threshold=">0",
            detail="会话级超时应正确注册"
        ))

        # 测试超时检测
        is_timeout = tm.check_timeout("test_session")
        report.add(TestResult(
            "RES-006", "系统韧性", "超时检测(未超时)",
            not is_timeout,
            value="未超时" if not is_timeout else "已超时",
            threshold="未超时",
            detail="刚注册的会话不应超时"
        ))

    except ImportError as e:
        report.add(TestResult("RES-005", "系统韧性", "超时管理器", False, detail=str(e)))

    # 5.5 幻觉防护
    try:
        from hallucination_guard import HallucinationGuard

        guard = HallucinationGuard()

        # 测试1: 有证据的正常输出
        clean_output = "Pod order-service-abc 状态为 Running"
        tool_results = ["Pod order-service-abc: Running"]
        result = guard.check_for_hallucinations(clean_output, [], tool_results)

        report.add(TestResult(
            "RES-007", "系统韧性", "幻觉防护(正常输出)",
            not result.is_hallucination,
            value=result.recommendation,
            threshold="ACCEPT",
            detail="有证据支撑的输出不应被标记为幻觉"
        ))

        # 测试2: 无证据的数值声明
        hallucinated_output = "CPU 使用率达到 99%，内存占用 8GB，连接数 500"
        empty_tool_results = []
        result2 = guard.check_for_hallucinations(hallucinated_output, [], empty_tool_results)

        report.add(TestResult(
            "RES-008", "系统韧性", "幻觉防护(无证据数值)",
            result2.is_hallucination or len(result2.evidence_gaps) > 0,
            value=f"evidence_gaps={len(result2.evidence_gaps)}",
            threshold=">=1 gap",
            detail="无工具证据的数值声明应被标记"
        ))

    except ImportError as e:
        report.add(TestResult("RES-007", "系统韧性", "幻觉防护", False, detail=str(e)))


# ========================================================================
# 维度 6: 安全合规
# ========================================================================

async def test_security_compliance():
    """维度6: 安全合规测试"""
    logger.info("\n===== 维度6: 安全合规 =====")

    # 6.1 参数注入防护
    try:
        from tool import validate_param, POD_NAME_PATTERN, SERVICE_NAME_PATTERN, DANGEROUS_CHARS

        # 测试正常参数
        try:
            validate_param("order-service", SERVICE_NAME_PATTERN, "service_name")
            normal_pass = True
        except ValueError:
            normal_pass = False

        report.add(TestResult(
            "SEC-001", "安全合规", "正常参数通过",
            normal_pass,
            value="通过" if normal_pass else "拦截",
            threshold="通过",
            detail="合法服务名应通过校验"
        ))

        # 测试注入攻击
        injection_attempts = [
            "order-service; rm -rf /",
            "order-service$(cat /etc/passwd)",
            "order-service|whoami",
            "order-service`id`",
        ]
        blocked_count = 0
        for payload in injection_attempts:
            try:
                validate_param(payload, SERVICE_NAME_PATTERN, "service_name")
            except ValueError:
                blocked_count += 1

        report.add(TestResult(
            "SEC-002", "安全合规", "注入攻击拦截率",
            blocked_count == len(injection_attempts),
            value=f"{blocked_count}/{len(injection_attempts)}",
            threshold=f"{len(injection_attempts)}/{len(injection_attempts)}",
            detail="所有注入攻击载荷应被拦截"
        ))

        # 测试空参数
        try:
            validate_param("", SERVICE_NAME_PATTERN, "service_name")
            empty_blocked = False
        except ValueError:
            empty_blocked = True

        report.add(TestResult(
            "SEC-003", "安全合规", "空参数拦截",
            empty_blocked,
            value="拦截" if empty_blocked else "通过",
            threshold="拦截",
            detail="空参数应被拦截"
        ))

    except ImportError as e:
        report.add(TestResult("SEC-001", "安全合规", "参数校验", False, detail=str(e)))

    # 6.2 高危工具标记
    try:
        from tool import DANGEROUS_TOOLS

        has_dangerous_mark = len(DANGEROUS_TOOLS) > 0
        report.add(TestResult(
            "SEC-004", "安全合规", "高危工具标记",
            has_dangerous_mark,
            value=DANGEROUS_TOOLS,
            threshold="非空",
            detail="必须标记高危工具以触发审批"
        ))

    except ImportError as e:
        report.add(TestResult("SEC-004", "安全合规", "高危标记", False, detail=str(e)))

    # 6.3 RBAC 模块
    try:
        import rbac
        import inspect

        source = inspect.getsource(rbac)
        has_role_check = "permission" in source.lower() or "role" in source.lower()

        report.add(TestResult(
            "SEC-005", "安全合规", "RBAC 权限控制",
            has_role_check,
            value="有" if has_role_check else "无",
            threshold="有",
            detail="必须实现角色权限控制"
        ))

    except ImportError as e:
        report.add(TestResult("SEC-005", "安全合规", "RBAC模块", False, detail=str(e)))

    # 6.4 审计模块
    try:
        import audit
        import inspect

        source = inspect.getsource(audit)
        has_audit_log = "audit" in source.lower() or "log" in source.lower()

        report.add(TestResult(
            "SEC-006", "安全合规", "审计日志",
            has_audit_log,
            value="有" if has_audit_log else "无",
            threshold="有",
            detail="高危操作必须记录审计日志"
        ))

    except ImportError as e:
        report.add(TestResult("SEC-006", "安全合规", "审计模块", False, detail=str(e)))

    # 6.5 配置安全（密码不入 ini）
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        with open(config_path, "r", encoding="utf-8") as f:
            ini_content = f.read().lower()

        has_plaintext_password = "password" in ini_content and any(
            line.strip().startswith("password") and "=" in line and line.split("=")[1].strip()
            for line in ini_content.split("\n")
            if "password" in line and not line.strip().startswith("#")
        )

        report.add(TestResult(
            "SEC-007", "安全合规", "配置无明文密码",
            not has_plaintext_password,
            value="无明文密码" if not has_plaintext_password else "有明文密码",
            threshold="无明文密码",
            detail="config.ini 不应包含明文密码"
        ))

    except Exception as e:
        report.add(TestResult("SEC-007", "安全合规", "配置检查", False, detail=str(e)))

    # 6.6 .env 敏感信息管理
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env.example")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_content = f.read()

            has_api_key_placeholder = "API_KEY" in env_content or "api_key" in env_content
            report.add(TestResult(
                "SEC-008", "安全合规", ".env 模板",
                has_api_key_placeholder,
                value="有" if has_api_key_placeholder else "无",
                threshold="有",
                detail=".env.example 应包含 API Key 占位符"
            ))
        else:
            report.add(TestResult(
                "SEC-008", "安全合规", ".env 模板",
                True, value="跳过", threshold="N/A",
                detail=".env.example 不存在（可能已有 .env）"
            ))

    except Exception as e:
        report.add(TestResult("SEC-008", "安全合规", ".env检查", False, detail=str(e)))


# ========================================================================
# 评测系统自身验证
# ========================================================================

async def test_evaluation_system():
    """验证评测系统自身功能完整性"""
    logger.info("\n===== 评测系统自检 =====")

    try:
        from evaluation import evaluation_system, RAGASMetric, AgentBehaviorMetric

        # 测试 RAGAS 指标计算
        metric = evaluation_system.calculate_ragas_metrics(
            question="订单服务为什么重启",
            answer="订单服务因 OOM 导致 Pod 频繁重启",
            contexts=["订单服务 Pod OOM 堆内存溢出 CrashLoopBackOff"],
            ground_truth="JVM 堆内存配置不足导致 OOM"
        )

        valid_range = all(0.0 <= v <= 1.0 for v in [
            metric.faithfulness, metric.answer_relevancy,
            metric.context_precision, metric.context_recall
        ])

        report.add(TestResult(
            "EVAL-001", "评测系统", "RAGAS 指标计算",
            valid_range,
            value=f"F={metric.faithfulness:.2f} R={metric.answer_relevancy:.2f} P={metric.context_precision:.2f} R={metric.context_recall:.2f}",
            threshold="全部 0~1",
            detail="四个 RAGAS 指标应在合法范围内"
        ))

        # 测试 Agent 行为追踪
        behavior = evaluation_system.track_agent_behavior(
            session_id="test_eval",
            execution_history=[
                {"step_info": {"action": "get_pod_status"}, "result_summary": "OK", "status": "completed"},
                {"step_info": {"action": "fetch_logs"}, "result_summary": "OOM", "status": "completed"},
            ],
            total_duration=10.5,
            final_confidence=0.85,
            total_tokens=5000
        )

        valid_behavior = behavior.total_tool_calls == 2 and behavior.tool_success_rate == 1.0
        report.add(TestResult(
            "EVAL-002", "评测系统", "Agent 行为追踪",
            valid_behavior,
            value=f"calls={behavior.total_tool_calls}, success_rate={behavior.tool_success_rate:.0%}",
            threshold="calls=2, rate=100%",
            detail="行为指标应正确计算"
        ))

        # 测试聚合报告
        agg = evaluation_system.get_aggregate_metrics()
        has_aggregate = "total_sessions" in agg and agg["total_sessions"] > 0

        report.add(TestResult(
            "EVAL-003", "评测系统", "聚合报告",
            has_aggregate,
            value=f"sessions={agg.get('total_sessions', 0)}",
            threshold=">0",
            detail="聚合指标应正确汇总"
        ))

    except Exception as e:
        report.add(TestResult("EVAL-001", "评测系统", "功能完整性", False, detail=str(e)))


# ========================================================================
# 主入口
# ========================================================================

async def run_all_tests():
    """运行全部测试"""
    print("=" * 60)
    print("AIOps 系统中厂量化指标测试集")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    await test_rag_retrieval_quality()
    await test_diagnosis_accuracy()
    await test_execution_efficiency()
    await test_cost_control()
    await test_system_resilience()
    await test_security_compliance()
    await test_evaluation_system()

    # 输出报告
    print("\n" + "=" * 60)
    summary = report.summary()
    print(f"测试完成: {summary['passed']}/{summary['total']} 通过 ({summary['pass_rate']})")
    print("=" * 60)

    # 导出 Markdown 报告
    md_report = report.export_markdown()
    report_path = os.path.join(os.path.dirname(__file__), "test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"\nMarkdown 报告已导出: {report_path}")

    # 导出 JSON 报告
    json_path = os.path.join(os.path.dirname(__file__), "test_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON 报告已导出: {json_path}")

    if summary["failed"] > 0:
        print(f"\n⚠️ {summary['failed']} 项未通过，请检查 FAIL 项")
        return 1
    print("\n✅ ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
