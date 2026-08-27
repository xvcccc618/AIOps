# context_assembler.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from bone import OpsAgentState, AgentStatus
from context_schema import (
    AgentContext,
    StaticContext,
    DynamicContext,
    EnvironmentContext,
    LoadMetric
)
from memory_manager import get_memory_manager
from prompt_registry import get_prompt_registry 

logger = logging.getLogger("ContextAssembler")


class ContextAssembler:
    """
    负责将 OpsAgentState 转换为 LLM 可理解的 AgentContext。
    集成短期记忆 (STM) 和长期记忆 (LTM)。
    """

    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.prompt_registry = get_prompt_registry()

    def assemble(self, state: OpsAgentState) -> AgentContext:
        """
        主入口：组装完整的上下文
        """
        # 1. 提取基础信息
        query = state.get("query", "")
        service_name = self._extract_service_hint(state)
        severity = state.get("incident_severity", "P3").value if hasattr(state.get("incident_severity"),
                                                                         'value') else str(
            state.get("incident_severity", "P3"))

        # 获取当前租户/命名空间，用于记忆隔离
        caller_namespace = state.get("extracted_entities", {}).get("tenant_id", "default_tenant")
        user_role = state.get("user_permission_level", "standard")

        # 2. 构建 Static Context (系统提示词、角色定义)
        static_ctx = self._build_static_context(service_name, severity, user_role)

        # 3. 构建 Dynamic Context (当前排查状态、STM)
        dynamic_ctx = self._build_dynamic_context(state)

        # 4. 构建 Environment Context (实时指标、LTM 经验)
        env_ctx = self._build_environment_context(state, service_name, caller_namespace)

        return AgentContext(
            static=static_ctx,
            dynamic=dynamic_ctx,
            environment=env_ctx,
            incident_severity=severity
        )

    def _extract_service_hint(self, state: OpsAgentState) -> str:
        """从 State 中提取服务名称"""
        entities = state.get("extracted_entities", {})
        return entities.get("service_name", "unknown_service")

    def _build_static_context(self, service_name: str, severity: str, user_role: str) -> StaticContext:
        """
        构建静态上下文：包含系统指令、角色定义、可用工具列表
        """
        # 获取基础 System Prompt
        system_prompt_template = self.prompt_registry.get("system_prompt_base")

        # 根据严重等级调整语气或指令
        if severity in ["P0", "P1"]:
            system_prompt_template += "\nIMPORTANT: This is a critical incident. Prioritize speed and accuracy. Avoid unnecessary chitchat."

        # 根据用户角色调整权限提示
        if user_role == "admin":
            system_prompt_template += "\nYou have ADMIN privileges. You can execute destructive operations if necessary."
        else:
            system_prompt_template += "\nYou have STANDARD privileges. Do NOT execute destructive operations without explicit confirmation."

        # 可用工具列表 (假设从 ToolRegistry 获取)
        available_tools = [
            {"name": "check_logs", "description": "Check application logs for errors"},
            {"name": "check_metrics", "description": "Check CPU/Memory/IO metrics"},
            {"name": "restart_service", "description": "Restart a specific service (Admin only)"},
            {"name": "query_db", "description": "Query database status"}
        ]

        return StaticContext(
            system_instruction=system_prompt_template,
            service_name=service_name,
            available_tools=available_tools,
            user_role=user_role
        )

    def _build_dynamic_context(self, state: OpsAgentState) -> DynamicContext:
        """
        构建动态上下文：包含当前查询、历史对话、STM (短期记忆)
        """
        query = state.get("query", "")
        conversation_history = state.get("conversation_history", [])

        # MemoryManager 内部已经处理了滑动窗口和摘要压缩
        stm_events = self.memory_manager.stm_events

        # 将 STM 事件格式化为字符串，注入到 Dynamic Context 中
        # 注意：这里只取最近的一部分，避免 Token 爆炸，因为 STM 头部已经有 Summary
        stm_snippets = []
        for event in stm_events[-10:]:  # 只取最近10个详细事件，前面的由 Summary 代表
            stamp = event.timestamp.strftime("%H:%M:%S")
            stm_snippets.append(f"[{stamp}] [{event.event_type}] {event.content}")

        stm_summary = ""
        # 如果第一个事件是 Summary，则提取出来作为高层摘要
        if stm_events and stm_events[0].event_type == "Summary":
            stm_summary = f"## Previous Investigation Summary\n{stm_events[0].content}\n"

        return DynamicContext(
            current_query=query,
            conversation_history=conversation_history,
            short_term_memory_summary=stm_summary,
            short_term_memory_recent_events="\n".join(stm_snippets),
            current_step=state.get("current_step", "initial_analysis")
        )

    def _build_environment_context(self, state: OpsAgentState, service_name: str,
                                   caller_namespace: str) -> EnvironmentContext:
        """
        构建环境上下文：包含实时负载指标、LTM (长期记忆) 经验
        """
        now = datetime.now(timezone.utc)

        # 1. 获取实时负载指标 (模拟调用监控系统)
        load_data = self._fetch_realtime_load(service_name)

        # 计算数据延迟
        captured_time_str = load_data.get("captured_at", now.isoformat())
        try:
            captured_time = datetime.fromisoformat(captured_time_str.replace('Z', '+00:00'))
            latency = (now - captured_time).total_seconds()
        except Exception:
            latency = 0.0

        load_metric = LoadMetric(
            cpu_percent=load_data.get("cpu", 0),
            memory_percent=load_data.get("memory", 0),
            io_wait=load_data.get("io_wait", 0),
            captured_at=captured_time_str,
            latency_seconds=latency
        )

        # 2.检索 LTM 经验 (带命名空间隔离和置信度评分)
        ltm_experiences = self.memory_manager.search_ltm(
            query=state.get("query", ""),
            caller_namespace=caller_namespace,
            service_hint=service_name
        )

        # 格式化 LTM 片段
        ltm_snippets = []
        if ltm_experiences:
            ltm_snippets.append("## Relevant Historical Experiences")
            for i, exp in enumerate(ltm_experiences, 1):
                snippet = (
                    f"### Case {i} (Confidence: {exp['confidence']:.2f})\n"
                    f"- **Symptom**: {exp['symptom']}\n"
                    f"- **Root Cause**: {exp['root_cause']}\n"
                    f"- **Resolution**: {exp['resolution']}\n"
                )
                ltm_snippets.append(snippet)
        else:
            ltm_snippets.append("## No relevant historical experiences found.")

        return EnvironmentContext(
            current_time=now.isoformat(),
            user_permission_level=state.get("user_permission_level", "standard"),
            load_metrics=load_metric,
            historical_memory_snippets=ltm_snippets
        )

    def _fetch_realtime_load(self, service_name: str) -> Dict[str, Any]:
        """
        模拟获取实时监控数据
        实际生产中应替换为对 Prometheus/Grafana API 的调用
        """
        import random
        # 模拟数据
        return {
            "cpu": random.uniform(10, 90),
            "memory": random.uniform(20, 80),
            "io_wait": random.uniform(0, 5),
            "captured_at": datetime.now(timezone.utc).isoformat()
        }


# 单例实例
_context_assembler_instance = ContextAssembler()


def get_context_assembler() -> ContextAssembler:
    return _context_assembler_instance