# prompt_registry.py
import os
import yaml
import time
import logging
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("PromptRegistry")


class PromptRegistry:
    def __init__(self, base_path: str = "./prompts"):
        self.base_path = Path(base_path)
        self._cache: Dict[str, Dict] = {}
        self._file_hashes: Dict[str, str] = {}
        self._default_system_prompt = "You are an expert SRE agent."

        self.base_path.mkdir(parents=True, exist_ok=True)
        self._init_default_files()

    def _init_default_files(self):
        generic_sop_path = self.base_path / "generic_sop.md"
        if not generic_sop_path.exists():
            generic_sop_path.write_text(
                "# Generic SOP\n1. Check Logs\n2. Check Metrics\n3. Restart if necessary"
            )

        system_prompt_path = self.base_path / "system_prompt.md"
        if not system_prompt_path.exists():
            system_prompt_path.write_text(
                "You are an expert SRE agent. Help users troubleshoot issues."
            )

    def _get_file_hash(self, file_path: Path) -> str:
        if not file_path.exists():
            return ""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_yaml_or_md(self, file_path: Path) -> Dict:
        if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        elif file_path.suffix == '.md':
            content = file_path.read_text(encoding='utf-8')
            return {"content": content}
        return {}

    def get(self, key: str, default: str = "") -> str:
        """通用 get 方法 — 兼容 context_assembler 的 prompt_registry.get('system_prompt_base') 调用"""
        if key in ("system_prompt", "system_prompt_base"):
            return self.get_system_prompt()
        # 尝试按 key 名加载对应文件
        for ext in (".md", ".yaml", ".yml"):
            path = self.base_path / f"{key}{ext}"
            if path.exists():
                return self._load_with_cache(path).get("content", default)
        return default

    def get_system_prompt(self) -> str:
        path = self.base_path / "system_prompt.md"
        return self._load_with_cache(path).get("content", self._default_system_prompt)

    def get_sops_for_service(self, service_name: str) -> List[str]:
        specific_path = self.base_path / f"{service_name}_sop.yaml"
        generic_path = self.base_path / "generic_sop.yaml"

        data = {}
        if specific_path.exists():
            data = self._load_with_cache(specific_path)
        elif generic_path.exists():
            data = self._load_with_cache(generic_path)

        if not data:
            specific_md = self.base_path / f"{service_name}_sop.md"
            generic_md = self.base_path / "generic_sop.md"
            if specific_md.exists():
                data = self._load_with_cache(specific_md)
            elif generic_md.exists():
                data = self._load_with_cache(generic_md)

        sops = data.get("sops", [])
        if not sops and "content" in data:
            sops = [data["content"]]

        return sops if isinstance(sops, list) else [str(sops)]

    def get_few_shots(self, service_name: str) -> List[Dict[str, str]]:
        specific_path = self.base_path / f"{service_name}_few_shots.yaml"
        if specific_path.exists():
            data = self._load_with_cache(specific_path)
            return data.get("few_shots", [])
        return []

    def _load_with_cache(self, file_path: Path) -> Dict:
        file_str = str(file_path)
        current_hash = self._get_file_hash(file_path)

        if file_str in self._cache and self._file_hashes.get(file_str) == current_hash:
            return self._cache[file_str]

        logger.info(f"Loading/Reloading prompt file: {file_path}")
        data = self._load_yaml_or_md(file_path)

        self._cache[file_str] = data
        self._file_hashes[file_str] = current_hash

        return data


_registry_instance = PromptRegistry()


def get_prompt_registry() -> PromptRegistry:
    return _registry_instance
