from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class ActionItemType(str, Enum):
    SHORT_TERM = "短期修复 (Immediate Fix)"
    MEDIUM_TERM = "中期优化 (Optimization)"
    LONG_TERM = "长期架构改造 (Architectural Refactor)"


class EvidenceItem(BaseModel):
    """证据链单元"""
    type: str = Field(description="证据类型: LOG_SNIPPET, METRIC_CHART, TOOL_OUTPUT, TRACE_ID")
    content: str = Field(description="证据内容：日志片段、监控链接、工具输出摘要或 TraceID")
    relevance: str = Field(description="该证据如何支持根因结论")


class ActionItem(BaseModel):
    """改进措施 Todo Item (SMART 原则)"""
    type: ActionItemType = Field(description="改进措施的类型")
    description: str = Field(
        description="具体的行动描述。必须包含具体动词和量化目标。例如：'将 JVM Heap 从 4G 增加到 8G' 而不是 '优化内存'")
    owner: str = Field(description="负责人，必须是具体的人或团队，严禁 'TBD'，如果未知则填 '需人工指定'")
    deadline: str = Field(description="截止日期，格式 YYYY-MM-DD。严禁 'TBD'，如果未知则填 '需人工指定'")
    priority: str = Field(description="优先级: P0/P1/P2")

    @validator('description')
    def check_smart_description(cls, v):
        if len(v) < 10:
            raise ValueError("Description too short, must be specific.")
        # 简单启发式检查：是否包含模糊词汇
        vague_words = ["优化", "加强", "注意", "尽量"]
        if any(word in v for word in vague_words):
            raise ValueError(f"Description contains vague words: {vague_words}. Must be SMART.")
        return v


class TimelineEvent(BaseModel):
    """故障时间线事件"""
    timestamp: str = Field(description="事件发生时间，格式 YYYY-MM-DD HH:MM:SS")
    event_type: str = Field(description="事件类型，如: Alert, Investigation, RootCauseFound, Recovery")
    description: str = Field(description="事件详细描述")


class RelatedArtifact(BaseModel):
    """关联资产"""
    type: str = Field(description="资产类型: Dashboard, LogQuery, Ticket, CodeCommit")
    name: str = Field(description="资产名称或简要描述")
    url_or_id: str = Field(description="链接地址或唯一ID")


class RCAResult(BaseModel):
    """完整的 RCA 报告模型 (增强证据链)"""
    incident_summary: str = Field(
        description="故障简述：包含故障发生时间、影响范围（受影响服务/用户）、核心现象（报错/延迟）。"
    )

    timeline: List[TimelineEvent] = Field(
        description="故障时间线：按时间顺序排列的关键事件列表。"
    )

    root_cause_analysis: str = Field(
        description="根因分析：使用 5 Whys 方法进行的深度推导过程。必须基于事实，逻辑清晰。"
    )

    # Q1: 证据链
    evidence_chain: List[EvidenceItem] = Field(
        description="证据链：支持根因结论的具体证据。每一个结论都必须有对应的日志、监控或工具输出作为支撑。无证据不结论。"
    )

    impact_assessment: str = Field(
        description="影响评估：受损用户量、资金损失预估、SLA 违约情况。如果 State 中缺乏具体数据，必须输出 '数据缺失，需人工补充'，严禁编造。"
    )

    action_items: List[ActionItem] = Field(
        description="改进 Todo List：分为短期、中期、长期，并指定 Owner 和 Deadline。必须符合 SMART 原则。"
    )

    related_artifacts: List[RelatedArtifact] = Field(
        default_factory=list,
        description="关联资产：相关的监控大盘链接、日志查询 DSL、历史相似工单 ID 等。"
    )

    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="AI 对此次 RCA 结论的置信度评分 (0-1)。"
    )