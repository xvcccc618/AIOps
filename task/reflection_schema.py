# reflection_schema.py
from pydantic import BaseModel, Field
from typing import List

class ReflectionResult(BaseModel):
    """反思结果数据结构，用于结构化评估当前排查状态"""
    current_hypothesis: str = Field(
        description="当前正在验证的假设（如：怀疑是 Redis 连接池打满）"
    )
    evidence_support_score: float = Field(
        ge=0.0, le=1.0,
        description="现有证据对该假设的支持度 (0.0 - 1.0)"
    )
    is_stuck: bool = Field(
        description="是否陷入死胡同。触发条件：连续重复动作、遇到权限拒绝或无效反馈"
    )
    critique: str = Field(
        description="LLM 对自身刚才一步操作的批评（如：我不该去查 Nginx 日志，因为报错明确是 DB 层的）"
    )
    next_strategy: str = Field(
        description="下一步的修正策略（如：放弃排查网络层，转向排查 DB 慢查询）"
    )
    # Q3 新增：强制要求 LLM 提供证据引用，用于后续代码交叉验证
    evidence_citations: List[str] = Field(
        default_factory=list,
        description="支撑你反思结论的客观证据摘要（如：'check_metrics 返回 CPU 99%'），用于交叉验证防甩锅"
    )