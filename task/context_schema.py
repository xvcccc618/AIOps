# context_schema.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class StaticContext(BaseModel):
    """静态上下文：System Prompt, SOP, Few-Shot"""
    system_prompt: str = Field(default="", description="基础系统提示词")
    system_instruction: str = Field(default="", description="系统指令 (兼容 context_assembler 写入)")
    service_name: str = Field(default="", description="目标服务名称")
    available_tools: List[Dict[str, str]] = Field(default_factory=list, description="可用工具列表")
    user_role: str = Field(default="standard", description="用户角色")
    sops: List[str] = Field(default_factory=list, description="当前服务相关的标准作业程序 (SOP)")
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list, description="Few-Shot 示例")


class DynamicContext(BaseModel):
    """动态上下文：Query, RAG, Tool Output, STM"""
    current_query: str = Field(default="", description="用户当前查询")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="对话历史")
    short_term_memory_summary: str = Field(default="", description="STM 压缩摘要")
    short_term_memory_recent_events: str = Field(default="", description="STM 最近事件文本")
    current_step: str = Field(default="initial_analysis", description="当前排查步骤标识")
    rag_results: List[Dict[str, Any]] = Field(default_factory=list, description="RAG 召回的历史 RCA 案例")
    tool_outputs: List[Dict[str, str]] = Field(default_factory=list, description="工具调用的关键输出")
    critic_warnings: List[str] = Field(default_factory=list, description="Critic 节点的强警告")


class LoadMetric(BaseModel):
    """带时间戳的负载指标"""
    cpu_percent: float
    memory_percent: float
    io_wait: float
    captured_at: str = Field(description="数据采集时间 ISO 格式")
    latency_seconds: float = Field(description="数据从采集到当前的延迟秒数", default=0.0)


class EnvironmentContext(BaseModel):
    """环境上下文：时间, 权限, 负载, 历史记忆"""
    current_time: str = Field(description="当前时间 ISO 格式")
    user_permission_level: str = Field(default="standard", description="报障用户权限级别")
    load_metrics: Optional[LoadMetric] = Field(None, description="目标机器实时负载")
    historical_memory_snippets: List[str] = Field(default_factory=list, description="近期历史记忆片段")


class AgentContext(BaseModel):
    """聚合后的完整上下文对象"""
    static: StaticContext
    dynamic: DynamicContext
    environment: EnvironmentContext

    # 元数据
    total_token_estimate: int = Field(default=0, description="预估总 Token 数")
    assembly_timestamp: datetime = Field(default_factory=datetime.now)
    incident_severity: str = Field(default="P3", description="故障等级")
