"""
Checkpointer 工厂：优先 AsyncRedisSaver（异步持久化，支持 HITL 挂起恢复），
Redis 不可用时快速失败并降级为 MemorySaver（内存版，会话不持久化）。
降级只影响断点恢复能力，不阻断主流程。

说明：主图全部使用 ainvoke 异步执行，必须使用 AsyncRedisSaver——
同步版 RedisSaver 未实现 aget_tuple，会在图执行时抛 NotImplementedError。
"""
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from settings import get_redis_config

logger = logging.getLogger("CheckpointFactory")

_PROBE_TIMEOUT_SECONDS = 2


def _probe_redis_sync(url: str) -> None:
    """同步 ping 探测：快速失败，避免编译出名义连接但实际不可用的 checkpointer"""
    import redis as redis_lib
    probe = redis_lib.Redis.from_url(url, socket_connect_timeout=_PROBE_TIMEOUT_SECONDS)
    try:
        probe.ping()
    finally:
        probe.close()


async def _probe_redis_async(url: str) -> None:
    """异步 ping 探测（与图执行路径一致）"""
    import redis.asyncio as aioredis
    probe = aioredis.from_url(url, socket_connect_timeout=_PROBE_TIMEOUT_SECONDS)
    try:
        await probe.ping()
    finally:
        await probe.aclose()


async def create_checkpointer(ini_path: str = "config.ini"):
    """异步版工厂：探测 → AsyncRedisSaver + asetup() → 失败降级 MemorySaver。

    因为 build_graph 需要 await 本函数，图构建入口也已改为 async。
    """
    try:
        url = get_redis_config(ini_path)["url"]
        await _probe_redis_async(url)
        saver = AsyncRedisSaver(redis_url=url)
        await saver.asetup()  # 首次运行创建 Redis 索引结构（幂等）
        logger.info("Checkpoint: AsyncRedisSaver (persistent)")
        return saver
    except Exception as e:
        logger.warning(
            f"Redis 不可用 ({type(e).__name__}: {e})。降级为 MemorySaver（无持久化，HITL 跨进程恢复将不可用）"
        )
        return MemorySaver()


def create_checkpointer_sync(ini_path: str = "config.ini"):
    """同步版工厂（仅供不需要实际执行图的场景：故障演练探测、类型检查）。

    返回 AsyncRedisSaver 实例但不执行 asetup()（协程需事件循环）。
    实际跑图请走 create_checkpointer()。
    """
    try:
        url = get_redis_config(ini_path)["url"]
        _probe_redis_sync(url)
        saver = AsyncRedisSaver(redis_url=url)
        logger.info("Checkpoint(sync probe): AsyncRedisSaver (not set up)")
        return saver
    except Exception as e:
        logger.warning(
            f"Redis 不可用 ({type(e).__name__}: {e})。降级为 MemorySaver（无持久化，HITL 跨进程恢复将不可用）"
        )
        return MemorySaver()
