import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

logger = logging.getLogger("HallucinationGuard")


class HallucinationCheckResult(BaseModel):
    """幻觉检测结果"""
    is_hallucination: bool = Field(description="是否检测到幻觉")
    confidence: float = Field(ge=0.0, le=1.0, description="检测置信度")
    suspicious_claims: List[str] = Field(default_factory=list, description="可疑声明")
    evidence_gaps: List[str] = Field(default_factory=list, description="证据缺口")
    recommendation: str = Field(description="建议：ACCEPT/VERIFY/REJECT")


class GlobalTimeoutManager:
    """全局超时管理器"""
    
    def __init__(self):
        self.session_timeouts: Dict[str, datetime] = {}
        self.default_timeout_minutes = 30  # 默认 30 分钟超时
        logger.info("Global timeout manager initialized")
    
    def register_session(self, session_id: str, timeout_minutes: Optional[int] = None):
        """注册会话超时时间"""
        timeout = timeout_minutes or self.default_timeout_minutes
        deadline = datetime.now() + timedelta(minutes=timeout)
        self.session_timeouts[session_id] = deadline
        logger.info(f"Session {session_id} registered with {timeout}min timeout")
    
    def check_timeout(self, session_id: str) -> bool:
        """检查会话是否超时"""
        if session_id not in self.session_timeouts:
            return False
        
        deadline = self.session_timeouts[session_id]
        is_timeout = datetime.now() > deadline
        
        if is_timeout:
            logger.warning(f"Session {session_id} has timed out!")
        
        return is_timeout
    
    def get_remaining_time(self, session_id: str) -> Optional[float]:
        """获取剩余时间（秒）"""
        if session_id not in self.session_timeouts:
            return None
        
        deadline = self.session_timeouts[session_id]
        remaining = (deadline - datetime.now()).total_seconds()
        return max(0, remaining)
    
    def extend_timeout(self, session_id: str, additional_minutes: int):
        """延长超时时间"""
        if session_id in self.session_timeouts:
            self.session_timeouts[session_id] += timedelta(minutes=additional_minutes)
            logger.info(f"Session {session_id} timeout extended by {additional_minutes}min")


class HallucinationGuard:
    """幻觉防护守卫"""
    
    def __init__(self):
        # 常见的幻觉模式
        self.hallucination_patterns = [
            "根据我的分析",
            "我认为",
            "可能是",
            "应该是",
            "理论上",
            "假设",
            "推测"
        ]
        
        # 需要验证的声明关键词
        self.claim_keywords = [
            "CPU 使用率",
            "内存占用",
            "错误日志",
            "连接数",
            "响应时间",
            "成功率"
        ]
        
        logger.info("Hallucination guard initialized")
    
    def check_for_hallucinations(
        self,
        agent_output: str,
        execution_history: List[Dict[str, Any]],
        tool_results: List[str]
    ) -> HallucinationCheckResult:
        """检查 Agent 输出是否包含幻觉"""
        
        suspicious_claims = []
        evidence_gaps = []
        
        # 1. 检查是否包含未经验证的具体数值
        agent_output_lower = agent_output.lower()
        
        for keyword in self.claim_keywords:
            if keyword in agent_output:
                # 检查这个声明是否有工具结果支持
                has_evidence = any(
                    keyword in result.lower()
                    for result in tool_results
                )
                if not has_evidence:
                    evidence_gaps.append(f"声明了'{keyword}'但无工具证据支持")
        
        # 2. 检查模糊性表达
        for pattern in self.hallucination_patterns:
            if pattern in agent_output:
                suspicious_claims.append(f"使用了模糊表达: '{pattern}'")
        
        # 3. 检查是否与执行历史矛盾
        for history in execution_history[-5:]:  # 检查最近 5 条
            result_summary = history.get("result_summary", "")
            # 如果 Agent 声称"正常"但工具返回了"错误"
            if "正常" in agent_output and "error" in result_summary.lower():
                suspicious_claims.append("声称正常但工具返回了错误")
                break
        
        # 4. 计算幻觉置信度
        suspicion_score = len(suspicious_claims) * 0.3 + len(evidence_gaps) * 0.4
        is_hallucination = suspicion_score > 0.5
        
        # 5. 生成建议
        if is_hallucination:
            recommendation = "REJECT"
        elif suspicious_claims or evidence_gaps:
            recommendation = "VERIFY"
        else:
            recommendation = "ACCEPT"
        
        result = HallucinationCheckResult(
            is_hallucination=is_hallucination,
            confidence=min(suspicion_score, 1.0),
            suspicious_claims=suspicious_claims,
            evidence_gaps=evidence_gaps,
            recommendation=recommendation
        )
        
        if is_hallucination:
            logger.warning(
                f"Hallucination detected! Suspicious: {len(suspicious_claims)}, "
                f"Gaps: {len(evidence_gaps)}"
            )
        
        return result
    
    def enforce_fact_checking(
        self,
        agent_output: str,
        check_result: HallucinationCheckResult
    ) -> str:
        """根据幻觉检查结果修正输出"""
        
        if check_result.recommendation == "REJECT":
            # 拒绝输出，要求重新验证
            return (
                f"检测到潜在幻觉，已拦截输出。\n"
                f"可疑点：\n" +
                "\n".join([f"- {s}" for s in check_result.suspicious_claims]) +
                "\n证据缺口：\n" +
                "\n".join([f"- {g}" for g in check_result.evidence_gaps]) +
                "\n\n请重新验证你的结论，确保所有声明都有工具证据支持。"
            )
        
        elif check_result.recommendation == "VERIFY":
            # 添加警告但允许输出
            warnings = []
            if check_result.suspicious_claims:
                warnings.append("注意：输出中包含模糊表达")
            if check_result.evidence_gaps:
                warnings.append("注意：部分声明缺少直接证据")
            
            return agent_output + "\n\n" + "\n".join(warnings)
        
        else:
            # ACCEPT: 直接返回
            return agent_output


# 全局实例
timeout_manager = GlobalTimeoutManager()
hallucination_guard = HallucinationGuard()


async def apply_timeout_to_task(task_coro, session_id: str):
    """为任务应用超时控制"""
    timeout_manager.register_session(session_id)
    
    try:
        # 使用 asyncio.wait_for 实现超时
        timeout_seconds = timeout_manager.get_remaining_time(session_id)
        if timeout_seconds is None:
            timeout_seconds = 1800  # 默认 30 分钟
        
        result = await asyncio.wait_for(task_coro, timeout=timeout_seconds)
        return result
    
    except asyncio.TimeoutError:
        logger.error(f"Task timed out for session {session_id}")
        return {
            "status": "TIMEOUT",
            "error": "任务执行超时，已强制终止",
            "session_id": session_id
        }


def check_output_hallucinations(
    output: str,
    execution_history: List[Dict[str, Any]],
    tool_results: List[str]
) -> HallucinationCheckResult:
    """便捷函数：检查输出幻觉"""
    return hallucination_guard.check_for_hallucinations(
        output, execution_history, tool_results
    )
