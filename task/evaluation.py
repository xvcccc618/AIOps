"""
实现 RAGAS 指标计算和 Agent 行为指标追踪
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from statistics import mean

logger = logging.getLogger("Evaluation")


class RAGASMetric(BaseModel):
    """RAGAS 评测指标"""
    faithfulness: float = Field(ge=0.0, le=1.0, description="忠实度：答案是否基于检索到的上下文")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="答案相关性：答案是否回答了问题")
    context_precision: float = Field(ge=0.0, le=1.0, description="上下文精确度：检索的文档是否相关")
    context_recall: float = Field(ge=0.0, le=1.0, description="上下文召回率：相关文档是否被检索到")


class AgentBehaviorMetric(BaseModel):
    """Agent 行为指标"""
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 效率指标
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_success_rate: float = 0.0
    
    # 时间指标
    total_duration_seconds: float = 0.0
    avg_tool_call_duration_seconds: float = 0.0
    
    # 路径效率
    replan_count: int = 0
    reflection_count: int = 0
    dead_ends_detected: int = 0
    
    # 资源消耗
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    
    # 结果质量
    final_confidence_score: float = 0.0
    user_satisfaction_rating: Optional[float] = None


class EvaluationSystem:
    """评测系统"""
    
    def __init__(self):
        self.session_metrics: List[AgentBehaviorMetric] = []
        self.ragas_scores: List[RAGASMetric] = []
        logger.info("Evaluation system initialized")
    
    def calculate_ragas_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGASMetric:
        """
        计算 RAGAS 指标
        注意：实际生产环境应调用 RAGAS 库或 LLM 进行计算
        这里提供简化的启发式计算
        """
        # 简化计算：基于关键词匹配和长度
        answer_lower = answer.lower()
        contexts_text = " ".join(contexts).lower()
        
        # 1. Faithfulness: 答案中的关键词是否在上下文中出现
        answer_words = set(answer_lower.split())
        context_words = set(contexts_text.split())
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            faithfulness = min(overlap * 1.5, 1.0)  # 放大系数
        else:
            faithfulness = 0.0
        
        # 2. Answer Relevancy: 答案是否包含问题关键词
        question_words = set(question.lower().split())
        if question_words and answer_words:
            relevance = len(question_words & answer_words) / len(question_words)
            answer_relevancy = min(relevance * 1.2, 1.0)
        else:
            answer_relevancy = 0.5
        
        # 3. Context Precision: 上下文是否包含答案关键词
        if contexts and answer_words:
            context_has_answer = sum(1 for w in answer_words if w in context_words) / len(answer_words)
            context_precision = min(context_has_answer * 1.3, 1.0)
        else:
            context_precision = 0.0
        
        # 4. Context Recall: 如果有 ground truth，检查上下文是否覆盖
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                recall = len(gt_words & context_words) / len(gt_words)
                context_recall = min(recall * 1.5, 1.0)
            else:
                context_recall = 0.5
        else:
            context_recall = 0.7  # 无 ground truth 时给默认值
        
        metric = RAGASMetric(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall
        )
        
        self.ragas_scores.append(metric)
        logger.info(f"RAGAS metrics calculated: {metric.model_dump()}")
        
        return metric
    
    def track_agent_behavior(
        self,
        session_id: str,
        execution_history: List[Dict[str, Any]],
        total_duration: float,
        final_confidence: float = 0.0,
        total_tokens: int = 0
    ) -> AgentBehaviorMetric:
        """追踪 Agent 行为指标"""
        total_calls = len(execution_history)
        successful = sum(1 for h in execution_history if h.get("status") == "completed")
        failed = total_calls - successful
        
        # 计算工具调用平均时长
        durations = [h.get("duration_seconds", 0) for h in execution_history]
        avg_duration = mean(durations) if durations else 0.0
        
        # 统计 replan 和 reflection
        replan_count = sum(1 for h in execution_history if "replan" in str(h).lower())
        reflection_count = sum(1 for h in execution_history if "reflection" in str(h).lower())
        
        # 估算成本 (简化模型：$0.0001 per token)
        estimated_cost = total_tokens * 0.0001 / 1000
        
        metric = AgentBehaviorMetric(
            session_id=session_id,
            total_tool_calls=total_calls,
            successful_tool_calls=successful,
            failed_tool_calls=failed,
            tool_success_rate=successful / total_calls if total_calls > 0 else 0.0,
            total_duration_seconds=total_duration,
            avg_tool_call_duration_seconds=avg_duration,
            replan_count=replan_count,
            reflection_count=reflection_count,
            total_tokens_used=total_tokens,
            estimated_cost_usd=estimated_cost,
            final_confidence_score=final_confidence
        )
        
        self.session_metrics.append(metric)
        logger.info(f"Agent behavior tracked for session {session_id}: {metric.model_dump()}")
        
        return metric
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """获取聚合指标（跨会话）"""
        if not self.session_metrics:
            return {"message": "No metrics available"}
        
        # 工具成功率
        success_rates = [m.tool_success_rate for m in self.session_metrics]
        avg_success_rate = mean(success_rates)
        
        # 平均时长
        durations = [m.total_duration_seconds for m in self.session_metrics]
        avg_duration = mean(durations)
        
        # 平均成本
        costs = [m.estimated_cost_usd for m in self.session_metrics]
        avg_cost = mean(costs)
        
        # RAGAS 平均分
        if self.ragas_scores:
            avg_faithfulness = mean([r.faithfulness for r in self.ragas_scores])
            avg_relevancy = mean([r.answer_relevancy for r in self.ragas_scores])
            avg_precision = mean([r.context_precision for r in self.ragas_scores])
            avg_recall = mean([r.context_recall for r in self.ragas_scores])
        else:
            avg_faithfulness = avg_relevancy = avg_precision = avg_recall = 0.0
        
        return {
            "total_sessions": len(self.session_metrics),
            "avg_tool_success_rate": avg_success_rate,
            "avg_session_duration_seconds": avg_duration,
            "avg_cost_usd": avg_cost,
            "ragas_scores": {
                "faithfulness": avg_faithfulness,
                "answer_relevancy": avg_relevancy,
                "context_precision": avg_precision,
                "context_recall": avg_recall
            }
        }
    
    def export_report(self) -> str:
        """导出评测报告（Markdown 格式）"""
        agg = self.get_aggregate_metrics()
        
        report = "# Agent 评测报告\n\n"
        report += f"生成时间：{datetime.now().isoformat()}\n\n"
        report += "## 概览\n\n"
        report += f"- 总会话数：{agg.get('total_sessions', 0)}\n"
        report += f"- 平均工具成功率：{agg.get('avg_tool_success_rate', 0):.2%}\n"
        report += f"- 平均会话时长：{agg.get('avg_session_duration_seconds', 0):.2f} 秒\n"
        report += f"- 平均成本：${agg.get('avg_cost_usd', 0):.4f}\n\n"
        
        report += "## RAGAS 指标\n\n"
        ragas = agg.get("ragas_scores", {})
        report += f"- 忠实度：{ragas.get('faithfulness', 0):.2f}\n"
        report += f"- 答案相关性：{ragas.get('answer_relevancy', 0):.2f}\n"
        report += f"- 上下文精确度：{ragas.get('context_precision', 0):.2f}\n"
        report += f"- 上下文召回率：{ragas.get('context_recall', 0):.2f}\n"
        
        return report


# 全局评测系统实例
evaluation_system = EvaluationSystem()
