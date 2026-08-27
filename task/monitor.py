"""
HITL 超时监控服务 (Watchdog)
利用 Redis TTL 机制检测超时的中断请求，并自动触发拒绝流程。
已从 config.py 迁移至 ini 配置读取。
"""
import asyncio
import json
import logging
import time
import configparser
import os
from typing import Optional

import redis.asyncio as aioredis
from langgraph.graph import StateGraph

logger = logging.getLogger("HITLMonitor")

# 默认配置文件路径
DEFAULT_INI_PATH = "config.ini"


def load_monitor_config(ini_path: str = DEFAULT_INI_PATH) -> dict:
    """
    从 ini 文件加载监控和 Redis 配置
    """
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        raise FileNotFoundError(f"Configuration file not found: {ini_path}")

    config.read(ini_path, encoding='utf-8')

    # Redis 连接：host/port/db 来自 ini，密码只来自 .env
    from settings import get_redis_config
    redis_url = get_redis_config(ini_path)["url"]

    # 读取超时配置 (优先从 [monitor] 节，其次从 [redis] 节，最后默认 300s)
    hitl_timeout = 300
    if config.has_section('monitor'):
        hitl_timeout = config.getint('monitor', 'hitl_timeout_seconds', fallback=300)
    elif config.has_option('redis', 'hitl_timeout_seconds'):
        hitl_timeout = config.getint('redis', 'hitl_timeout_seconds', fallback=300)

    # （Redis URL 已由 settings.get_redis_config 提供）

    return {
        "redis_url": redis_url,
        "hitl_timeout_seconds": hitl_timeout
    }


class HITLTimeoutMonitor:
    def __init__(self, graph: StateGraph, redis_client: aioredis.Redis, hitl_timeout_seconds: int = 300):
        self.graph = graph
        self.redis = redis_client
        self.is_running = False
        self.prefix = "langgraph:pending_approval:"
        self.hitl_timeout_seconds = hitl_timeout_seconds

    async def start(self, check_interval: int = 10):
        """启动监控循环"""
        self.is_running = True
        logger.info(
            f"[Monitor] HITL 监控已启动。检查间隔: {check_interval}s, SLA Timeout: {self.hitl_timeout_seconds}s")

        while self.is_running:
            try:
                await self._process_timeouts()
            except Exception as e:
                logger.error(f"[Monitor] 监控循环异常: {e}", exc_info=True)

            await asyncio.sleep(check_interval)

    async def stop(self):
        self.is_running = False
        logger.info(" [Monitor] HITL 监控已停止")

    async def register_pending_approval(self, thread_id: str, interrupt_id: str):
        """
        当 Graph 触发 interrupt 时调用此方法。
        在 Redis 中注册一个带 TTL 的 Key。
        """
        key = f"{self.prefix}{thread_id}:{interrupt_id}"
        value = json.dumps({
            "thread_id": thread_id,
            "interrupt_id": interrupt_id,
            "created_at": time.time()
        })
        # 设置 Key，过期时间为 SLA 时间
        await self.redis.setex(key, self.hitl_timeout_seconds, value)
        logger.debug(f"[Monitor] 注册待审批任务: {key}, TTL: {self.hitl_timeout_seconds}s")

    async def _process_timeouts(self):
        """
        扫描 Redis 中已过期的 Key。
        """
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=f"{self.prefix}*", count=100)

            for key in keys:
                # 检查剩余生存时间
                ttl = await self.redis.ttl(key)

                # 如果 TTL <= 0，说明已过期或即将过期
                if ttl <= 0:
                    # 再次确认 Key 是否还存在（防止竞态条件）
                    exists = await self.redis.exists(key)
                    if not exists:
                        # Key 已被 Redis 自动删除，说明已超时
                        # 从 Key 名称中解析 thread_id 和 interrupt_id
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        parts = key_str.replace(self.prefix, '').split(':')
                        if len(parts) == 2:
                            thread_id, interrupt_id = parts
                            logger.warning(f"[Monitor] 检测到超时任务: Thread={thread_id}, Interrupt={interrupt_id}")

                            # 触发自动拒绝
                            await self._trigger_auto_reject(thread_id, interrupt_id)

                            # 确保 Key 被清理
                            await self.redis.delete(key)

            if cursor == 0:
                break

    async def _trigger_auto_reject(self, thread_id: str, interrupt_id: str):
        """
        向指定的 Thread 注入拒绝指令，唤醒挂起的 Graph。
        """
        try:
            logger.info(f"[Monitor] 正在对 Thread {thread_id} 执行自动拒绝...")

            input_data = {
                "__interrupt__": [{
                    "id": interrupt_id,
                    "response": {
                        "approved": False,
                        "reason": f"SLA Timeout: 超过 {self.hitl_timeout_seconds} 秒未响应"
                    }
                }]
            }

            config_dict = {"configurable": {"thread_id": thread_id}}

            await self.graph.ainvoke(input_data, config=config_dict)

            logger.info(f"[Monitor] Thread {thread_id} 自动拒绝完成，Graph 已恢复运行。")

        except Exception as e:
            logger.error(f"[Monitor] 自动拒绝失败 Thread {thread_id}: {e}", exc_info=True)


# 全局单例 Monitor
monitor_instance: Optional[HITLTimeoutMonitor] = None


# 辅助函数：创建并初始化 Monitor
async def create_monitor(graph: StateGraph, ini_path: str = DEFAULT_INI_PATH) -> HITLTimeoutMonitor:
    """
    工厂函数：从 ini 加载配置，创建 Redis 客户端和 Monitor 实例
    """
    conf = load_monitor_config(ini_path)
    redis_client = aioredis.from_url(conf["redis_url"])

    monitor = HITLTimeoutMonitor(
        graph=graph,
        redis_client=redis_client,
        hitl_timeout_seconds=conf["hitl_timeout_seconds"]
    )
    return monitor