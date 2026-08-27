"""
RCA 知识库（Milvus 版）
- 向量库：Milvus（database=itcast, collection=edurag_final），支持增量写入与真实标量过滤
- Embedding：远程 BGE-M3（SiliconFlow /v1/embeddings, 1024 维），无本地路径依赖
- 图引导过滤：入库时自动从案例文本提取拓扑组件名写入 components 字段，
  检索时用 `components like "%X%"` 布尔过滤表达式真实收窄候选（不再是占位宽过滤）
- 对外接口保持不变：search_with_parent_mapping / search_with_parent_mapping_and_filter /
  get_parent_text / add_case / get_rca_kb
- Schema 迁移：检测到旧集合缺少 components 字段时 drop 重建（样例数据可由代码自动回灌）
- 降级原则：Milvus 或远程 Embedding 不可用时，检索返回空列表（走 NO_RELEVANT_HISTORY），不阻断主流程
"""
import logging
import hashlib
from typing import List, Dict, Optional, Tuple

import requests
from pydantic import BaseModel

from settings import get_milvus_config, get_siliconflow_config
from topology import get_topology_graph

logger = logging.getLogger("RCA_KB")

PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
CHILD_OVERLAP = 50
EMBED_BATCH_SIZE = 32
EMBED_TIMEOUT_SECONDS = 30


class RCACase(BaseModel):
    case_id: str
    symptom: str
    root_cause: str
    resolution: str
    architecture_version: str
    is_deprecated: bool = False
    metrics_chart_url: Optional[str] = None
    topology_graph_data: Optional[Dict] = None


class RemoteEmbedder:
    """远程 BGE-M3 Embedding（SiliconFlow 兼容 /v1/embeddings），支持批量"""

    def __init__(self):
        cfg = get_siliconflow_config()
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["embedding_model"]
        self.available = bool(self.api_key) and not self.api_key.startswith("sk-your-")
        if not self.available:
            logger.warning("未配置有效的 SILICONFLOW_API_KEY，Embedding 不可用，知识库检索将返回空结果")
        else:
            logger.info(f"Remote Embedder ready: {self.model}")

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.available or not texts:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        results: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    headers=headers,
                    timeout=EMBED_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                results.extend([d["embedding"] for d in data])
            except Exception as e:
                logger.error(f"Embedding API failed: {type(e).__name__}: {str(e)[:200]}")
                return None
        return results


def _collect_topology_component_names() -> set:
    """收集拓扑图中所有可过滤的组件名：图节点 + 边上的 target_db"""
    topo = get_topology_graph()
    names = set(topo.graph.nodes())
    for _, _, data in topo.graph.edges(data=True):
        target_db = data.get("target_db")
        if target_db:
            names.add(target_db)
    names.discard("")
    return names


def extract_components_from_text(text: str) -> str:
    """从案例文本中提取出现过的拓扑组件名，'|' 分隔（用于 components 字段与 like 过滤）"""
    if not text:
        return ""
    found = sorted(name for name in _collect_topology_component_names() if name in text)
    return "|".join(found)


class RCAKnowledgeBase:
    def __init__(self):
        self.parent_store: Dict[str, RCACase] = {}
        self.parent_text_map: Dict[str, str] = {}
        self.child_to_parent_map: Dict[str, str] = {}
        self.milvus_client = None
        self.database_name: str = ""
        self.collection_name: str = ""
        self.embedder = RemoteEmbedder()
        self._init_milvus()
        self._load_sample_data()
        self._sync_index()

    # ---------- Milvus 初始化 ----------
    def _create_collection(self):
        """建库建集合（含 components 标量过滤字段）"""
        from pymilvus import DataType
        cfg = get_milvus_config()
        dim = cfg["dim"]

        if self.database_name not in self.milvus_client.list_databases():
            self.milvus_client.create_database(self.database_name)
            logger.info(f"Milvus database created: {self.database_name}")

        schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("child_hash", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=2000)
        schema.add_field("case_id", DataType.VARCHAR, max_length=64)
        schema.add_field("arch", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=4000)
        # 图引导过滤字段：入库时从文本提取的拓扑组件名（'|' 分隔），支持 like 布尔过滤
        schema.add_field("components", DataType.VARCHAR, max_length=1024, default_value="")

        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            database_name=self.database_name,
        )
        logger.info(f"Milvus collection created: {self.database_name}.{self.collection_name} (dim={dim}, with components field)")

    def _init_milvus(self):
        cfg = get_milvus_config()
        self.database_name = cfg["database"]
        self.collection_name = cfg["collection"]
        try:
            from pymilvus import MilvusClient
            uri = f"http://{cfg['host']}:{cfg['port']}"
            self.milvus_client = MilvusClient(uri=uri)

            if not self.milvus_client.has_collection(self.collection_name, database_name=self.database_name):
                self._create_collection()
            else:
                # Schema 迁移：旧集合缺少 components 字段时 drop 重建
                # （旧集合中仅有可由代码自动回灌的样例数据，重建成本可控）
                desc = self.milvus_client.describe_collection(self.collection_name, database_name=self.database_name)
                field_names = {f.get("name") for f in desc.get("fields", [])}
                if "components" not in field_names:
                    logger.warning(
                        f"Collection {self.collection_name} lacks 'components' field (legacy schema). "
                        f"Dropping and recreating for graph-guided filtering support."
                    )
                    self.milvus_client.drop_collection(self.collection_name, database_name=self.database_name)
                    self._create_collection()
                else:
                    logger.info(f"Milvus collection exists: {self.database_name}.{self.collection_name}")
        except Exception as e:
            logger.error(f"Milvus init failed: {type(e).__name__}: {e}. 检索将降级为空结果。")
            self.milvus_client = None

    # ---------- 分块 ----------
    def _split_into_chunks(self, text: str) -> List[Tuple[str, str, str]]:
        """返回 [(child_text, child_hash, parent_hash)]，同时维护 parent_text_map"""
        chunks = []
        for i in range(0, len(text), PARENT_CHUNK_SIZE - CHILD_OVERLAP):
            parent_text = text[i:i + PARENT_CHUNK_SIZE]
            if not parent_text.strip():
                continue
            parent_hash = hashlib.md5(parent_text.encode("utf-8")).hexdigest()
            self.parent_text_map[parent_hash] = parent_text
            for j in range(0, len(parent_text), CHILD_CHUNK_SIZE - CHILD_OVERLAP):
                child_text = parent_text[j:j + CHILD_CHUNK_SIZE]
                if child_text.strip():
                    child_hash = hashlib.md5(child_text.encode("utf-8")).hexdigest()
                    self.child_to_parent_map[child_hash] = parent_hash
                    chunks.append((child_text, child_hash, parent_hash))
        return chunks

    def _load_sample_data(self):
        samples = [
            RCACase(
                case_id="RCA-001",
                symptom="订单服务 Pod 频繁重启，状态为 CrashLoopBackOff，日志出现 java.lang.OutOfMemoryError: Java heap space",
                root_cause="应用代码中本地缓存未设置最大容量和过期策略，导致堆内存溢出。",
                resolution="1. 紧急重启服务恢复；2. 代码修复：引入 Caffeine 缓存并设置 maxSize；3. 增加 JVM Heap 监控告警。",
                architecture_version="java-spring-boot-v2.7",
            ),
            RCACase(
                case_id="RCA-003",
                symptom="网关 P99 延迟从 50ms 上升至 2s，部分请求返回 504 Gateway Timeout，追踪发现下游用户服务响应极慢。",
                root_cause="下游用户服务因大对象分配导致频繁 Full GC，STW (Stop-The-World) 时间过长，拖垮上游网关。",
                resolution="1. 调整用户服务 JVM 参数，增大 Young Gen 比例；2. 优化大对象序列化逻辑；3. 网关层增加熔断降级策略。",
                architecture_version="java-spring-boot-v3.0",
            ),
        ]
        for case in samples:
            self.parent_store[case.case_id] = case

    # ---------- 索引同步（增量） ----------
    def _sync_index(self):
        if not self.milvus_client or not self.embedder.available:
            logger.warning("Milvus 或 Embedder 不可用，跳过索引同步")
            return
        rows = []
        for case in self.parent_store.values():
            full_text = f"Symptom: {case.symptom}\nRoot Cause: {case.root_cause}\nResolution: {case.resolution}"
            components = extract_components_from_text(full_text)
            chunks = self._split_into_chunks(full_text)
            for child_text, child_hash, parent_hash in chunks:
                rows.append({
                    "child_hash": child_hash,
                    "text": child_text,
                    "case_id": case.case_id,
                    "arch": case.architecture_version,
                    "parent_hash": parent_hash,
                    "parent_text": self.parent_text_map[parent_hash],
                    "components": components,
                })
        if not rows:
            return
        # 跳过已存在的 chunk（重复启动不重复写）
        try:
            ids = [r["child_hash"] for r in rows]
            id_filter = ", ".join([f'"{i}"' for i in ids])
            existed = self.milvus_client.query(
                self.collection_name, filter=f"child_hash in [{id_filter}]",
                output_fields=["child_hash"], database_name=self.database_name,
            )
            existed_ids = {r["child_hash"] for r in existed}
            new_rows = [r for r in rows if r["child_hash"] not in existed_ids]
            if not new_rows:
                logger.info("Milvus index already up to date.")
                return
            texts = [r["text"] for r in new_rows]
            vectors = self.embedder.embed(texts)
            if vectors is None:
                logger.error("Embedding failed, skip index sync")
                return
            for r, v in zip(new_rows, vectors):
                r["vector"] = v
            self.milvus_client.insert(self.collection_name, data=new_rows, database_name=self.database_name)
            logger.info(f"Milvus index synced: {len(new_rows)} new chunks inserted.")
        except Exception as e:
            logger.error(f"Milvus sync failed: {type(e).__name__}: {e}")

    # ---------- 对外接口 ----------
    def add_case(self, case: RCACase):
        """增量写入新案例（Milvus 优势：无需重建整个索引）"""
        logger.info(f"[KB] Adding new case: {case.case_id}")
        self.parent_store[case.case_id] = case
        if not self.milvus_client or not self.embedder.available:
            logger.warning("Milvus/Embedder 不可用，新案例仅暂存内存")
            return
        try:
            full_text = f"Symptom: {case.symptom}\nRoot Cause: {case.root_cause}\nResolution: {case.resolution}"
            components = extract_components_from_text(full_text)
            chunks = self._split_into_chunks(full_text)
            rows = [{
                "child_hash": child_hash,
                "text": child_text,
                "case_id": case.case_id,
                "arch": case.architecture_version,
                "parent_hash": parent_hash,
                "parent_text": self.parent_text_map[parent_hash],
                "components": components,
            } for child_text, child_hash, parent_hash in chunks]
            vectors = self.embedder.embed([r["text"] for r in rows])
            if vectors is None:
                logger.error("Embedding failed, case not indexed")
                return
            for r, v in zip(rows, vectors):
                r["vector"] = v
            self.milvus_client.insert(self.collection_name, data=rows, database_name=self.database_name)
            logger.info(f"[KB] Case {case.case_id} indexed with {len(rows)} chunks (components={components or 'none'}).")
        except Exception as e:
            logger.error(f"add_case failed: {type(e).__name__}: {e}")

    def search_with_parent_mapping(self, query: str, k: int = 10) -> List[Dict]:
        if not self.milvus_client:
            return []
        try:
            q_vec = self.embedder.embed([query])
            if q_vec is None:
                return []
            results = self.milvus_client.search(
                self.collection_name,
                data=q_vec,
                limit=k,
                output_fields=["text", "case_id", "arch", "parent_hash", "parent_text", "child_hash", "components"],
                database_name=self.database_name,
            )
            return self._format_hits(results)
        except Exception as e:
            logger.error(f"Search error: {type(e).__name__}: {e}")
            return []

    def search_with_parent_mapping_and_filter(self, query: str, allowed_components: List[str], k: int = 10) -> List[Dict]:
        """图引导检索：用 Milvus 布尔过滤表达式按 components 字段真实收窄候选。

        allowed_components 来自拓扑图的下游依赖（如 TiDB-B），
        过滤表达式形如：components like "%TiDB-B%" or components like "%Payment-Service%"。
        过滤后无结果时返回空列表（语义上正确：该组件确实没有历史案例），由检索节点回退到纯语义路。
        """
        if not self.milvus_client:
            return []
        if not allowed_components:
            return self.search_with_parent_mapping(query, k=k)
        try:
            q_vec = self.embedder.embed([query])
            if q_vec is None:
                return []
            # 组件名来自拓扑图（非用户输入），仍做基本转义防注入过滤表达式
            clauses = [
                f'components like "%{str(c).replace(chr(34), "")}%"'
                for c in allowed_components if str(c).strip()
            ]
            if not clauses:
                return self.search_with_parent_mapping(query, k=k)
            filter_expr = " or ".join(clauses)
            logger.info(f"[KB] Graph-guided filter: {filter_expr}")
            results = self.milvus_client.search(
                self.collection_name,
                data=q_vec,
                limit=k,
                filter=filter_expr,
                output_fields=["text", "case_id", "arch", "parent_hash", "parent_text", "child_hash", "components"],
                database_name=self.database_name,
            )
            return self._format_hits(results)
        except Exception as e:
            logger.error(f"Filtered search error: {type(e).__name__}: {e}. Falling back to semantic search.")
            return self.search_with_parent_mapping(query, k=k)

    def _format_hits(self, results) -> List[Dict]:
        formatted = []
        if not results:
            return formatted
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                parent_hash = entity.get("parent_hash", "")
                parent_text = entity.get("parent_text", "")
                if parent_hash and parent_text:
                    self.parent_text_map[parent_hash] = parent_text
                formatted.append({
                    "doc": {
                        "id": entity.get("child_hash"),
                        "content": entity.get("text", ""),
                        "metadata": {
                            "case_id": entity.get("case_id", ""),
                            "arch": entity.get("arch", ""),
                            "parent_hash": parent_hash,
                            "child_hash": entity.get("child_hash", ""),
                            "components": entity.get("components", ""),
                        },
                    },
                    "score": float(hit.get("distance", 0.0)),
                    "parent_hash": parent_hash,
                    "parent_text": parent_text,
                })
        return formatted

    def get_parent_text(self, parent_hash: str) -> str:
        if parent_hash in self.parent_text_map:
            return self.parent_text_map[parent_hash]
        # 内存未命中时回查 Milvus
        if not self.milvus_client:
            return ""
        try:
            rows = self.milvus_client.query(
                self.collection_name,
                filter=f'parent_hash == "{parent_hash}"',
                output_fields=["parent_text"],
                limit=1,
                database_name=self.database_name,
            )
            if rows:
                text = rows[0].get("parent_text", "")
                if text:
                    self.parent_text_map[parent_hash] = text
                return text
        except Exception as e:
            logger.error(f"get_parent_text failed: {e}")
        return ""


_kb_instance = None


def get_rca_kb() -> RCAKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = RCAKnowledgeBase()
    return _kb_instance
