# utils.py
import asyncio
import time
import logging
from functools import wraps
from typing import Type, Tuple, Optional, Dict, Any
import random

logger = logging.getLogger("utils")

# Python 类型 -> JSON Schema 类型映射（用于把 {字段: 类型} 简写字典转成合法 JSON Schema）
_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def to_structured_output_schema(schema: Dict[str, Any], title: str = "StructuredOutput") -> dict:
    """把 {字段名: Python类型} 简写字典转换为合法 JSON Schema。

    新版 langchain-core 的 with_structured_output 要求 dict 必须是
    带顶层 'title' 键的完整 JSON Schema，否则抛 'Unsupported function'。
    支持 str/int/float/bool/List[...]；其他类型回退为 string。
    """
    import typing
    properties: Dict[str, Any] = {}
    for name, typ in schema.items():
        origin = getattr(typ, "__origin__", None)
        if origin is list or origin is typing.List:
            inner = typ.__args__[0] if getattr(typ, "__args__", None) else str
            inner_type = _PY_TO_JSON_TYPE.get(inner, "string")
            properties[name] = {"type": "array", "items": {"type": inner_type}}
        elif origin is not None:
            # Optional[...] 等其他泛型：回退为 string
            properties[name] = {"type": "string"}
        else:
            properties[name] = {"type": _PY_TO_JSON_TYPE.get(typ, "string")}
    return {
        "title": title,
        "type": "object",
        "properties": properties,
    }


def pydantic_model_to_json_schema(model_cls) -> dict:
    """把 Pydantic 模型转换为 $defs 全部内联展开的 JSON Schema。

    用于 with_structured_output(method="json_mode")：当前 LLM 端点不支持
    function-calling 的 response_format，且对含 $ref/$defs 的 JSON Schema
    可能解析不佳，这里把引用全部解引用并内联为扁平 schema。
    返回的是 (schema, model_cls) 中可直接传给 method="json_mode" 的 schema。
    """
    schema = model_cls.model_json_schema()

    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                target = defs.get(ref_name, {})
                return _resolve(dict(target))  # 展开引用
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    resolved = _resolve(schema)
    resolved.setdefault("title", model_cls.__name__)
    return resolved


def retry_with_backoff(
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    异步函数重试装饰器，支持指数退避。
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # 计算延迟: min(base_delay * 2^attempt, max_delay)
                        delay = random.uniform(0, min(max_delay, base_delay * (2 ** attempt)))
                        logger.warning(
                            f"[Retry] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"[Retry] {func.__name__} failed after {max_retries + 1} attempts.")

            raise last_exception

        return wrapper

    return decorator


class CircuitBreaker:
    """
    简单的内存级熔断器。
    状态: CLOSED (正常) -> OPEN (熔断) -> HALF_OPEN (尝试恢复)
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"[CircuitBreaker] State changed to OPEN due to {self.failure_count} failures.")

    def is_open(self) -> bool:
        if self.state == "CLOSED":
            return False

        # 检查是否过了恢复时间
        if time.time() - self.last_failure_time > self.recovery_timeout:
            logger.info("[CircuitBreaker] Recovery timeout reached, moving to HALF_OPEN (allowing one try).")
            self.state = "HALF_OPEN"
            return False  # Allow one try

        return True  # Still OPEN


# 全局熔断器注册表: tool_name -> CircuitBreaker
GLOBAL_CIRCUIT_BREAKERS = {}


def get_circuit_breaker(tool_name: str) -> CircuitBreaker:
    if tool_name not in GLOBAL_CIRCUIT_BREAKERS:
        GLOBAL_CIRCUIT_BREAKERS[tool_name] = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    return GLOBAL_CIRCUIT_BREAKERS[tool_name]
