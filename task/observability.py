"""
可观测性模块：
1. structlog 结构化日志 —— 兼容现有 logging.getLogger，无需改动各模块代码
2. LLM 调用指标采集（次数 / token / 耗时 / 错误）——通过 LangGraph config["callbacks"] 注入，
   汇总后接入 evaluation_system.track_agent_behavior
"""
import sys
import time
import logging

import structlog
from langchain_core.callbacks import BaseCallbackHandler

_LOG_CONFIGURED = False


def setup_logging(level: int = logging.INFO, json_output: bool = False):
    """把 stdlib logging 桥接到 structlog 渲染；幂等，可重复调用"""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    # 1) stdlib logging -> structlog ProcessorFormatter（现有模块无需改动）
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=[structlog.stdlib.add_log_level, timestamper],
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 压制三方库噪音
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 2) structlog 原生 logger（新代码可用 get_logger）
    structlog.configure(
        processors=[structlog.processors.add_log_level, timestamper, renderer],
        logger_factory=structlog.WriteLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    _LOG_CONFIGURED = True


def get_logger(name: str = None):
    return structlog.get_logger(name)


class LLMetricsHandler(BaseCallbackHandler):
    """
    LLM 调用指标采集器。
    用法：config = {"configurable": {...}, "callbacks": [LLMetricsHandler()]}
    结束后 handler.summary() 得到 token/耗时统计，传给 evaluation_system。
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.llm_calls = 0
        self.llm_errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_time_seconds = 0.0
        self._start_times = {}

    # ---- 开始 ----
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start_times[run_id] = time.time()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start_times[run_id] = time.time()

    # ---- 结束 ----
    def on_llm_end(self, response, *, run_id, **kwargs):
        self.llm_calls += 1
        start = self._start_times.pop(run_id, None)
        if start is not None:
            self.llm_time_seconds += time.time() - start
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.llm_errors += 1
        self._start_times.pop(run_id, None)

    def summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_time_seconds": round(self.llm_time_seconds, 2),
        }
