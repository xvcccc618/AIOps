# memory_manager.py (Updated extract_and_save_experience method)
import uuid
import logging
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from bone import OpsAgentState
from memory_schema import MemoryEvent, ExperienceItem, ExperienceQuality

logger = logging.getLogger("MemoryManager")


class SecureVectorDB:
    def __init__(self):
        self.store: Dict[str, List[ExperienceItem]] = {}

    def add(self, item: ExperienceItem):
        ns = item.namespace
        if ns not in self.store:
            self.store[ns] = []
        existing_ids = {exp.id for exp in self.store[ns]}
        if item.id not in existing_ids:
            self.store[ns].append(item)
            logger.info(f"LTM [NS:{ns}]: Added experience {item.id}")

    def search(self, namespace: str, query_embedding: List[float], top_k: int = 5) -> List[ExperienceItem]:
        if namespace not in self.store:
            return []
        return self.store[namespace][:20]


_vdb_instance = SecureVectorDB()


class MemoryManager:
    def __init__(self, stm_max_events: int = 20, stm_recent_keep: int = 10):
        self.stm_max_events = stm_max_events
        self.stm_recent_keep = stm_recent_keep
        self.stm_events: List[MemoryEvent] = []
        self.vdb = _vdb_instance

    def add_stm(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        entities = self._extract_key_entities(content)
        event = MemoryEvent(
            event_type=event_type,
            content=content,
            metadata=metadata or {},
            key_entities=entities
        )
        self.stm_events.append(event)

        recent_non_summary_count = sum(1 for e in self.stm_events if e.event_type != "Summary")
        has_summary = any(e.event_type == "Summary" for e in self.stm_events)

        if (has_summary and recent_non_summary_count > self.stm_recent_keep) or \
                (len(self.stm_events) > self.stm_max_events):
            self.compact_stm()

    def _extract_key_entities(self, text: str) -> List[str]:
        entities = []
        ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
        entities.extend(ips)
        codes = re.findall(r'\b[E|HTTP]\d{3,4}\b', text)
        entities.extend(codes)
        return list(set(entities))

    def compact_stm(self):
        if len(self.stm_events) <= self.stm_recent_keep + 1:
            return

        recents = [e for e in self.stm_events if e.event_type != "Summary"]
        events_to_keep_raw = recents[-self.stm_recent_keep:]
        events_to_summarize = [e for e in self.stm_events if e not in events_to_keep_raw]

        if not events_to_summarize:
            return

        all_original_entities = set()
        for e in events_to_summarize:
            all_original_entities.update(e.key_entities)

        raw_summary = self._generate_summary_llm(events_to_summarize)
        validated_summary = self._validate_summary_facts(raw_summary, all_original_entities)

        summary_event = MemoryEvent(
            event_type="Summary",
            content=validated_summary,
            metadata={"original_count": len(events_to_summarize), "compressed_at": datetime.now().isoformat()},
            key_entities=list(all_original_entities)
        )

        self.stm_events = [summary_event] + events_to_keep_raw
        logger.info(f"STM: Compacted. New total: {len(self.stm_events)} (1 Summary + {len(events_to_keep_raw)} Recent)")

    def _generate_summary_llm(self, events: List[MemoryEvent]) -> str:
        lines = [f"[{e.event_type}] {e.content}" for e in events]
        return f"Summary of {len(events)} steps: " + "; ".join(lines[-3:])

    def _validate_summary_facts(self, summary: str, original_entities: Set[str]) -> str:
        summary_entities = set(self._extract_key_entities(summary))
        new_entities = summary_entities - original_entities
        if new_entities:
            return f"Conservative Summary: Involved {len(original_entities)} entities."
        return summary

    def extract_and_save_experience(self, state: OpsAgentState, rca_report: Dict[str, Any], caller_namespace: str):
        try:
            symptom = rca_report.get("incident_summary", "Unknown")
            root_cause = rca_report.get("root_cause_analysis", "Unknown")

            # 修复：兼容字典和对象两种格式
            action_items_raw = rca_report.get("action_items", [])
            actions = []
            for item in action_items_raw:
                if isinstance(item, dict):
                    actions.append(item.get("description", ""))
                else:
                    # 假设是对象
                    actions.append(getattr(item, 'description', str(item)))

            service = state.get("extracted_entities", {}).get("service_name", "generic")
            severity = state.get("incident_severity", "P3")
            acl_tags = set(state.get("system_hints", []))
            if not acl_tags: acl_tags.add("internal")

            experience = ExperienceItem(
                id=str(uuid.uuid4()),
                namespace=caller_namespace,
                acl_tags=acl_tags,
                symptom=symptom,
                root_cause=root_cause,
                resolution_steps=actions,
                service_name=service,
                severity=severity,
                quality=ExperienceQuality(success_count=1, fail_count=0, last_used_at=datetime.now())
            )
            self.vdb.add(experience)
            logger.info(f"[LTM] 提取经验成功: '{symptom}' -> '{root_cause}'")
        except Exception as e:
            logger.error(f"Failed to save experience: {e}")
            raise e

    def search_ltm(self, query: str, caller_namespace: str, service_hint: str = None) -> List[Dict[str, Any]]:
        candidates = self.vdb.search(namespace=caller_namespace, query_embedding=[], top_k=10)
        scored_candidates = []
        now = datetime.now()
        for exp in candidates:
            if service_hint and exp.service_name != service_hint: continue
            similarity_score = 0.9
            confidence = exp.quality.confidence_score
            recency = exp.quality.recency_factor
            final_score = similarity_score * (0.6 * confidence + 0.4 * recency)
            exp.quality.last_used_at = now
            scored_candidates.append((final_score, exp))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, exp in scored_candidates[:3]:
            results.append({
                "symptom": exp.symptom,
                "root_cause": exp.root_cause,
                "resolution": "\n".join(exp.resolution_steps),
                "service": exp.service_name,
                "confidence": exp.quality.confidence_score,
                "score": round(score, 2)
            })
        return results


_memory_manager_instance = MemoryManager()


def get_memory_manager() -> MemoryManager:
    return _memory_manager_instance