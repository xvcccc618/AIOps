"""
集中配置管理
原则：非敏感配置（host/port/db、超时阈值）留在 config.ini；
     敏感信息（密码、API Key）只存在于 .env / 环境变量，绝不入库。
"""
import os
import logging
import configparser
from pathlib import Path

logger = logging.getLogger("Settings")

_TASK_DIR = Path(__file__).resolve().parent
_ENV_PATH = _TASK_DIR / ".env"

# .env 只加载一次：load_dotenv 默认不覆盖已存在的环境变量，
# 因此显式设置的 os.environ 值始终优先于 .env 文件值。
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    logger.warning("python-dotenv 未安装，仅从进程环境变量读取配置")


def _env(key: str, default=None):
    """实时读取环境变量（不缓存快照，支持运行中注入/变更）"""
    return os.getenv(key, default)


def get_redis_config(ini_path: str = "config.ini") -> dict:
    """读取 Redis 连接配置：host/port/db 来自 ini，密码只来自环境变量"""
    ini = _TASK_DIR / ini_path
    if not ini.exists():
        raise FileNotFoundError(f"Config not found: {ini}")
    config = configparser.ConfigParser()
    config.read(ini, encoding="utf-8")
    redis_section = config["redis"]
    host = redis_section.get("host", "localhost")
    port = redis_section.getint("port", 6379)
    db = redis_section.getint("db", 0)

    password = os.getenv("REDIS_PASSWORD")
    if not password:
        # 兼容回退：ini 里仍有 password 时给出迁移警告
        password = redis_section.get("password", None)
        if password:
            logger.warning("检测到 config.ini 中仍有密码，请尽快迁移到 .env 的 REDIS_PASSWORD")

    auth_part = f":{password}@" if password else ""
    return {"url": f"redis://{auth_part}{host}:{port}/{db}"}


def get_llm_config() -> dict:
    """读取 LLM 配置（OpenAI 兼容接口，如 DeepSeek）"""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    }


def get_model_paths() -> dict:
    """本地模型路径：优先环境变量，默认指向项目内模型目录"""
    return {
        "embedding": _env("EMBEDDING_MODEL_PATH", str(_TASK_DIR / "all-MiniLM-L6-v2")),
        "reranker": _env("RERANKER_MODEL_PATH", str(_TASK_DIR / "ms-marco-MiniLM-L-6-v2")),
    }


def get_siliconflow_config() -> dict:
    """远程 BGE 服务（SiliconFlow 兼容端点）：embedding + rerank 均走 API，无本地路径依赖"""
    return {
        "api_key": _env("SILICONFLOW_API_KEY"),
        "base_url": _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "embedding_model": _env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "rerank_model": _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    }


def get_milvus_config(ini_path: str = "config.ini") -> dict:
    """读取 Milvus 连接配置（均为非敏感项）"""
    ini = _TASK_DIR / ini_path
    if not ini.exists():
        raise FileNotFoundError(f"Config not found: {ini}")
    config = configparser.ConfigParser()
    config.read(ini, encoding="utf-8")
    section = config["milvus"] if config.has_section("milvus") else {}
    return {
        "host": section.get("host", "localhost"),
        "port": section.get("port", "19530"),
        "database": section.get("database_name", "itcast"),
        "collection": section.get("collection_name", "edurag_final"),
        "dim": int(section.get("dim", "1024")),
    }


def _read_ini(ini_path: str = "config.ini") -> configparser.ConfigParser:
    ini = _TASK_DIR / ini_path
    if not ini.exists():
        raise FileNotFoundError(f"Config not found: {ini}")
    config = configparser.ConfigParser()
    config.read(ini, encoding="utf-8")
    return config


def get_db_config(ini_path: str = "config.ini") -> dict:
    """MySQL 连接配置：host/port/库名来自 ini，账号密码只来自环境变量"""
    config = _read_ini(ini_path)
    section = config["mysql"] if config.has_section("mysql") else {}
    return {
        "host": section.get("host", "localhost"),
        "port": section.getint("port", 3306),
        "database": section.get("database", "aiops"),
        "user": _env("MYSQL_USER", "root"),
        "password": _env("MYSQL_PASSWORD", ""),
    }


def get_prometheus_config(ini_path: str = "config.ini") -> dict:
    """Prometheus HTTP API 地址（非敏感项）"""
    config = _read_ini(ini_path)
    section = config["prometheus"] if config.has_section("prometheus") else {}
    host = section.get("host", "localhost")
    port = section.getint("port", 9090)
    return {"base_url": section.get("base_url", f"http://{host}:{port}")}


def get_k8s_config() -> dict:
    """K8s 连接配置：默认读 kubeconfig；可用 K8S_API_SERVER 显式指定"""
    return {
        "api_server": _env("K8S_API_SERVER"),          # None = 走 kubeconfig
        "token": _env("K8S_TOKEN"),                    # 可选 bearer token
        "namespace": _env("K8S_NAMESPACE", "default"),
    }
