# memory_schema.py
from typing import List, Optional, Dict, Any, Literal, Set
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryEventType(str):
    ACTION = "Action"
    OBSERVATION = "Observation"
    THOUGHT = "Thought"
    SUMMARY = "Summary"


class MemoryEvent(BaseModel):
    """短期记忆单元"""
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["Action", "Observation", "Thought", "Summary"] = Field(description="事件类型")
    content: str = Field(description="事件具体内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    key_entities: List[str] = Field(default_factory=list, description="从内容中提取的关键实体，用于校验幻觉")


class ExperienceQuality(BaseModel):
    """经验质量与衰减指标"""
    success_count: int = Field(default=0, description="被验证成功的次数")
    fail_count: int = Field(default=0, description="被验证失败的次数")
    last_used_at: Optional[datetime] = Field(None, description="最后一次被检索/使用的时间")
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def confidence_score(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # 默认中性置信度
        return self.success_count / total

    @property
    def recency_factor(self) -> float:
        if not self.last_used_at:
            return 0.5
        days_since = (datetime.now() - self.last_used_at).days
        # 指数衰减：每过30天，权重减半
        return max(0.1, 0.5 ** (days_since / 30.0))


class ExperienceItem(BaseModel):
    """长期记忆经验单元"""
    id: str = Field(description="唯一ID")
    namespace: str = Field(description="Q3: 命名空间/租户ID，用于隔离")
    acl_tags: Set[str] = Field(default_factory=set, description="Q3: 访问控制标签，如 'internal', 'public'")

    symptom: str = Field(description="故障现象")
    root_cause: str = Field(description="根因")
    resolution_steps: List[str] = Field(description="解决步骤")
    service_name: str = Field(description="涉及服务")
    severity: str = Field(description="故障等级")

    quality: ExperienceQuality = Field(default_factory=ExperienceQuality, description="Q2: 质量指标")

    # 向量嵌入通常存储在外部，这里保留引用
    embedding_id: Optional[str] = Field(None, description="向量数据库中的文档ID")


class MemoryState(BaseModel):
    short_term_events: List[MemoryEvent] = Field(default_factory=list)
    last_compact_time: Optional[datetime] = None