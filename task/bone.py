import uuid
import time
import logging
from enum import Enum
from typing import TypedDict, Annotated, Union, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

logger = logging.getLogger("Bone")


def merge_list(existing: list, new: Union[list, Any, None]) -> list:
    if new is None:
        return existing
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    else:
        return existing + [new]


class AgentStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = 'SUCCESS'
    FALLBACK = 'FALLBACK'
    FAILED = 'FAILED'
    HANDOFF = 'HANDOFF'  # 用于触发 Agent 间的交接


class IncidentSeverity(str, Enum):
    P0 = "P0_Critical"
    P1 = "P1_High"
    P2 = "P2_Medium"
    P3 = "P3_Low"
    P4 = "P4_Trivial"


def parse_incident_severity(severity: Any) -> "IncidentSeverity":
    """把用户输入的严重等级（如 "P0"/"P1_High"/"p2"）健壮解析为 IncidentSeverity。

    直接 IncidentSeverity(f"{severity}_Medium") 仅在 P2 时成立，其余等级会抛 ValueError。
    解析失败时回退 P2_Medium（中间档，预算与话术均安全）。
    """
    candidate = str(severity or "").strip().upper()
    if not candidate:
        return IncidentSeverity.P2
    for level in IncidentSeverity:
        if candidate == level.value.upper() or candidate == level.name:
            return level
    # 只给了 "P0" 这类裸等级：按等级补默认后缀
    suffix_map = {"P0": "Critical", "P1": "High", "P2": "Medium", "P3": "Low", "P4": "Trivial"}
    if candidate in suffix_map:
        return IncidentSeverity(f"{candidate}_{suffix_map[candidate]}")
    return IncidentSeverity.P2


# ================= 多 Agent 协作相关定义 =================

class SpecialistRole(str, Enum):
    SUPERVISOR = "supervisor"
    L1_AGENT = "l1_agent"
    L2_AGENT = "l2_agent"
    DBA_AGENT = "dba_agent"


def parse_specialist_role(raw: Any) -> Optional["SpecialistRole"]:
    """把 LLM 输出的角色字符串解析为 SpecialistRole。

    Prompt 要求输出大写枚举名（如 "L2_AGENT"），而枚举值是小写（"l2_agent"），
    直接 SpecialistRole(raw) 会抛 ValueError。此助手同时匹配枚举名与枚举值，
    大小写不敏感，解析失败返回 None 由调用方兜底。
    """
    if raw is None:
        return None
    candidate = str(raw).strip().lower()
    if not candidate:
        return None
    for role in SpecialistRole:
        if candidate == role.value or candidate == role.name.lower():
            return role
    return None


class AgentHandoff(BaseModel):
    """Agent 交接请求数据结构 (增加已排除项)"""
    from_agent: SpecialistRole = Field(description="发起交接的 Agent")
    to_agent: SpecialistRole = Field(description="目标接收 Agent")
    context_summary: str = Field(description="交接的上下文摘要，包含已排查的线索和当前卡点")
    reason: str = Field(description="发起交接的具体原因")
    excluded_hypotheses: List[str] = Field(
        default_factory=list,
        description="发起方已经排查并排除的假设，防止接收方重复排查"
    )


class MultiAgentSupervisorState(TypedDict, total=False):
    """Supervisor 全局状态"""
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    current_specialist: SpecialistRole
    handoff_request: Optional[AgentHandoff]
    specialist_findings: Dict[str, str]
    final_answer: str
    status: AgentStatus
    incident_severity: IncidentSeverity
    extracted_entities: Dict[str, str]
    topology_context: str
    error_log: Annotated[list[dict], merge_list]

    handoff_count: int
    handoff_history: List[AgentHandoff]
    is_arbitrating: bool

    # 内部传递字段
    _task_for_specialist: str
    _excluded_hypotheses: List[str]


class SpecialistState(TypedDict, total=False):
    """Specialist 局部状态"""
    task_description: str
    context_from_supervisor: str
    messages: Annotated[list[BaseMessage], add_messages]
    specialist_findings: str
    handoff_request: Optional[AgentHandoff]
    status: AgentStatus
    local_execution_history: Annotated[list[dict], merge_list]
    excluded_hypotheses: List[str]


class ScratchpadEntry(BaseModel):
    agent_role: str
    finding: str
    timestamp: float = Field(default_factory=time.time)


@dataclass
class Document:
    text: str
    score: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionItemType(str, Enum):
    SHORT_TERM = "短期修复 (Immediate Fix)"
    MEDIUM_TERM = "中期优化 (Optimization)"
    LONG_TERM = "长期架构改造 (Architectural Refactor)"


class EvidenceItem(BaseModel):
    type: str = Field(description="证据类型")
    content: str = Field(description="证据内容")
    relevance: str = Field(description="相关性")


class ActionItem(BaseModel):
    type: ActionItemType = Field(description="类型")
    description: str = Field(description="描述")
    owner: str = Field(description="负责人")
    deadline: str = Field(description="截止日期")
    priority: str = Field(description="优先级")


class TimelineEvent(BaseModel):
    timestamp: str = Field(description="时间")
    event_type: str = Field(description="类型")
    description: str = Field(description="描述")


class RelatedArtifact(BaseModel):
    type: str = Field(description="类型")
    name: str = Field(description="名称")
    url_or_id: str = Field(description="链接")


class RCAResult(BaseModel):
    incident_summary: str = Field(description="简述")
    timeline: List[TimelineEvent] = Field(description="时间线")
    root_cause_analysis: str = Field(description="根因")
    evidence_chain: List[EvidenceItem] = Field(description="证据链")
    impact_assessment: str = Field(description="影响")
    action_items: List[ActionItem] = Field(description="改进")
    related_artifacts: List[RelatedArtifact] = Field(default_factory=list, description="资产")
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")


class RCACase(BaseModel):
    case_id: str
    symptom: str
    root_cause: str
    resolution: str
    architecture_version: str
    is_deprecated: bool = False
    metrics_chart_url: Optional[str] = Field(None, description="URL")
    topology_graph_data: Optional[Dict] = Field(None, description="拓扑")


# ================= 核心 State 定义 (所有节点依赖) =================

class OpsAgentState(TypedDict, total=False):
    """主 Agent 全局状态 — 所有单 Agent 节点共享的核心状态类型"""
    # --- 基础输入 ---
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    incident_severity: IncidentSeverity
    status: AgentStatus

    # --- 路由与规划 ---
    next_action: str
    current_plan: List[Dict[str, Any]]
    current_step: int
    approved_tools: List[str]

    # --- RAG / 检索 ---
    expanded_search_queries: List[str]
    related_components_for_filter: List[str]
    valid_mapped_entities: List[str]
    retrieved_context: List[str]
    final_answer: str

    # --- 执行 ---
    execution_history: Annotated[list[dict], merge_list]
    pending_tool_calls: List[Dict[str, Any]]
    available_tools: List[Any]

    # --- 拓扑 ---
    topology_context: str
    extracted_entities: Dict[str, str]

    # --- 反思与纠错 ---
    failed_paths: List[str]
    next_strategy_hint: str
    reflection_count: int
    reset_count: int                      # 反思策略重置计数，≥3 升级 HITL
    replan_count: int                     # executor 触发重规划计数，≥3 转兜底
    reflection_result: Dict[str, Any]

    # --- Critic ---
    critic_decision: Dict[str, Any]

    # --- 记忆与权限 ---
    system_hints: Annotated[list[str], merge_list]
    user_permission_level: str
    conversation_history: List[Dict[str, str]]

    # --- 错误 ---
    error_log: Annotated[list[dict], merge_list]

    # --- 多 Agent 桥接 ---
    handoff_request: Optional[AgentHandoff]
    specialist_findings: Dict[str, str]


class ExpertState(TypedDict, total=False):
    """Expert Agent 子图状态 (experts.py 使用)"""
    task_description: str
    context_from_supervisor: str
    messages: Annotated[list[BaseMessage], add_messages]
    specialist_findings: str
    handoff_request: Optional[AgentHandoff]
    status: AgentStatus
    local_execution_history: Annotated[list[dict], merge_list]
    excluded_hypotheses: List[str]
    tool_call_count: int
    role: str
