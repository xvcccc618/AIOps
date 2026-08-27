"""
Reranker 服务：远程 BGE 版（SiliconFlow /v1/rerank）
- 替代原本地 CrossEncoder(ms-marco) 实现，无本地模型路径依赖，可直接进 Linux 容器
- 批处理打分，推理异常时降级 Mock（关键词重叠模拟），保证检索链路不中断
"""
import logging
from typing import List, Dict, Any

import requests

from settings import get_siliconflow_config

logger = logging.getLogger("RerankerService")

RERANK_TIMEOUT_SECONDS = 15


class RerankerService:
    """
    BGE 远程重排服务：
    1. 调用 SiliconFlow 兼容的 /v1/rerank 接口（模型 bge-reranker-v2-m3）
    2. 单次批量打分（API 原生支持 documents 数组）
    3. API Key 缺失或调用异常时降级为 Mock 关键词打分，链路不中断
    """

    def __init__(self):
        cfg = get_siliconflow_config()
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["rerank_model"]
        self.is_mock = False
        if not self.api_key or self.api_key.startswith("sk-your-"):
            logger.warning("未配置有效的 SILICONFLOW_API_KEY，Reranker 降级为 Mock 模式")
            self.is_mock = True
        else:
            logger.info(f"BGE Reranker ready (remote): {self.model}")

    def compute_scores(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        计算相关性得分
        :param query: 用户查询
        :param candidates: 候选列表，每个元素包含 'text' 字段
        :return: 得分列表（0~1，与候选顺序对齐）
        """
        if not candidates:
            return []

        if self.is_mock:
            return self._mock_compute_scores(query, candidates)

        documents = [c["text"] for c in candidates]
        url = f"{self.base_url}/rerank"
        try:
            resp = requests.post(
                url,
                json={"model": self.model, "query": query, "documents": documents, "top_n": len(documents)},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=RERANK_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            logger.error(f"Remote rerank failed: {type(e).__name__}: {str(e)[:200]}. Switching to Mock mode.")
            self.is_mock = True
            return self._mock_compute_scores(query, candidates)

        # results 是 [{index, relevance_score}]，需要按原始顺序还原
        scores = [0.0] * len(documents)
        for item in payload.get("results", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores

    def _mock_compute_scores(self, query: str, candidates: List[Dict]) -> List[float]:
        """Mock 逻辑：关键词重叠度模拟相关性（仅降级兜底用）"""
        scores = []
        query_lower = query.lower()
        for c in candidates:
            text = c.get("text", "").lower()
            match_count = sum(1 for w in query_lower.split() if len(w) > 2 and w in text)
            scores.append(min(0.9, 0.1 + match_count * 0.2))
        return scores


# 全局单例
_reranker_instance: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance
