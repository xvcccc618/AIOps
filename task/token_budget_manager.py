import re
import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple, Union
from bone import OpsAgentState, AgentStatus, IncidentSeverity
import json
logger = logging.getLogger("RAGRetrievalNode")

TOP_K_PER_PATH = 10
K_RRF = 60
SENTENCE_END_REGEX = re.compile(r'[.;?!\n]')

try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except:
    class FakeEncoder:
        def encode(self, text): return [0] * (len(text) // 4)

        def decode(self, tokens): return ""


    ENCODER = FakeEncoder()

BASE_BUDGETS = {
    "P0_Critical": 32000,
    "P1_High": 16000,
    "P2_Medium": 8000,
    "P3_Low": 4000,
    "P4_Trivial": 2000
}

STATIC_RESERVATION_RATIO = 0.2


class TokenBudgetManager:

    @staticmethod
    def count_tokens(text: str) -> int:
        if not text: return 0
        return len(ENCODER.encode(text))

    @staticmethod
    def get_dynamic_budget(severity: Union[str, IncidentSeverity]) -> int:
        if isinstance(severity, IncidentSeverity):
            sev_value = severity.value
        else:
            sev_value = severity
        return BASE_BUDGETS.get(sev_value, BASE_BUDGETS["P3_Low"])

    @staticmethod
    def apply_budget(context: Any) -> Any:
        """
        全局预算裁剪。
        【优化】如果 RAG 节点已经输出了严格受控的上下文（通过长度校验），则跳过二次截断。
        """
        severity = getattr(context, 'incident_severity', 'P3')
        max_tokens = TokenBudgetManager.get_dynamic_budget(severity)
        static_reservation = int(max_tokens * STATIC_RESERVATION_RATIO)
        dynamic_budget = max_tokens - static_reservation

        remaining_budget = dynamic_budget

        # 1. Critical Context
        query_text = getattr(getattr(context, 'dynamic', None), 'current_query', '')
        warnings = getattr(getattr(context, 'dynamic', None), 'critic_warnings', [])

        query_tokens = TokenBudgetManager.count_tokens(query_text)
        warning_tokens = sum(TokenBudgetManager.count_tokens(w) for w in warnings)
        critical_usage = query_tokens + warning_tokens

        if critical_usage > remaining_budget:
            if hasattr(context, 'dynamic'): context.dynamic.critic_warnings = []
            remaining_budget -= query_tokens
        else:
            remaining_budget -= critical_usage

        # 2. Tool Outputs (Compress)
        tool_outputs = getattr(getattr(context, 'dynamic', None), 'tool_outputs', [])
        trimmed_tool_outputs = []
        # 预留 40% 给 RAG 作为安全缓冲，其余给工具
        tool_budget = remaining_budget * 0.6

        for tool_out in tool_outputs:
            output_text = tool_out.get('output', '')
            original_tokens = TokenBudgetManager.count_tokens(output_text)
            if original_tokens > tool_budget:
                compressed = TokenBudgetManager._compress_log_output(output_text, max_tokens=int(tool_budget))
                trimmed_tool_outputs.append({**tool_out, 'output': compressed})
                tool_budget -= TokenBudgetManager.count_tokens(compressed)
            else:
                trimmed_tool_outputs.append(tool_out)
                tool_budget -= original_tokens
            if tool_budget <= 0: break

        if hasattr(context, 'dynamic'): context.dynamic.tool_outputs = trimmed_tool_outputs
        remaining_budget = tool_budget + (remaining_budget * 0.4)  # 回收未使用的工具预算给 RAG

        # 3. RAG Documents (Optimized: Skip if already within budget)
        rag_results = getattr(getattr(context, 'dynamic', None), 'rag_results', [])

        # 计算当前 RAG 总 Token
        current_rag_tokens = sum(TokenBudgetManager.count_tokens(r.get('content', '')) for r in rag_results)

        final_rag = []
        if current_rag_tokens <= remaining_budget:
            # 【关键优化】如果已经在预算内，直接保留，避免二次破坏性截断
            final_rag = rag_results
            remaining_budget -= current_rag_tokens
        else:
            # 降级策略：按分数过滤
            logger.warning("[Global Budget] RAG exceeds budget. Applying fallback filtering.")
            sorted_rag = sorted(rag_results, key=lambda x: x.get('score', 0), reverse=True)
            for rag_item in sorted_rag:
                content = rag_item.get('content', '')
                tokens = TokenBudgetManager.count_tokens(content)
                if tokens <= remaining_budget:
                    final_rag.append(rag_item)
                    remaining_budget -= tokens
                else:
                    truncated = TokenBudgetManager._truncate_text(content, remaining_budget)
                    if truncated:
                        final_rag.append({**rag_item, 'content': truncated})
                        remaining_budget -= TokenBudgetManager.count_tokens(truncated)
                if remaining_budget <= 0: break

        if hasattr(context, 'dynamic'): context.dynamic.rag_results = final_rag
        if hasattr(context, 'total_token_estimate'):
            context.total_token_estimate = max_tokens - remaining_budget

        return context

    @staticmethod
    def apply_budget_to_rag_context(
            raw_items: List[Dict[str, Any]],
            severity: Union[str, IncidentSeverity]
    ) -> List[str]:
        """
        高级 RAG 预算裁剪。
        :param raw_items: List of dicts with keys: 'child_content', 'parent_text', 'score'
        :param severity: Incident severity
        :return: List of trimmed text strings
        """
        if not raw_items:
            return []

        limit = TokenBudgetManager.get_dynamic_budget(severity)
        rag_limit = int(limit * 0.6)

        # 检查是否触发“放弃父块”策略 (Condition B: Total Parent Length > Limit)
        total_parent_tokens = 0
        for item in raw_items:
            if item.get('parent_text'):
                total_parent_tokens += TokenBudgetManager.count_tokens(item['parent_text'])

        use_smart_extension = total_parent_tokens > rag_limit

        logger.info(
            f"[RAG Budget] Mode: {'Smart Extension' if use_smart_extension else 'Parent Return'}, Limit: {rag_limit}")

        if not use_smart_extension:
            # 简单模式：直接返回父块（假设父块本身不大，或者数量少）
            results = []
            seen_parents = set()
            for item in raw_items:
                p_text = item.get('parent_text')
                if p_text and id(p_text) not in seen_parents:  # 简单去重
                    results.append(p_text)
                    seen_parents.add(id(p_text))
            return results

        # --- Smart Extension Mode ---
        total_score = sum(item['score'] for item in raw_items)
        if total_score == 0:
            weights = [1.0 / len(raw_items)] * len(raw_items)
        else:
            weights = [item['score'] / total_score for item in raw_items]

        final_texts = []
        for i, item in enumerate(raw_items):
            child_content = item['child_content']
            parent_text = item['parent_text']
            score = item['score']

            if not parent_text:
                # 无父块，直接截断子块
                budget = max(int(rag_limit * weights[i]), 200)
                final_texts.append(TokenBudgetManager._truncate_text_smart(child_content, budget))
                continue

            trimmed = TokenBudgetManager._smart_extend_chunk(
                child_content=child_content,
                parent_text=parent_text,
                budget=int(rag_limit * weights[i]),
                min_budget=200
            )
            if trimmed:
                final_texts.append(trimmed)

        return final_texts

    @staticmethod
    def _smart_extend_chunk(
            child_content: str,
            parent_text: str,
            budget: int,
            min_budget: int = 200
    ) -> str:
        """
        核心逻辑：在父块中定位子块，上下延伸，正则截断。
        """
        budget = max(budget, min_budget)

        # 1. 定位子块
        start_idx = parent_text.find(child_content)
        if start_idx == -1:
            # 尝试短匹配
            short_child = child_content[:50]
            start_idx = parent_text.find(short_child)

        if start_idx == -1:
            return TokenBudgetManager._truncate_text_smart(child_content, budget)

        # 2. 计算延伸范围 (Token -> Char 近似转换: 1 token ~= 3 chars for mixed CN/EN)
        char_budget = budget * 3
        half_char_budget = char_budget // 2

        ext_start = max(0, start_idx - half_char_budget)
        ext_end = min(len(parent_text), start_idx + len(child_content) + half_char_budget)

        segment = parent_text[ext_start:ext_end]

        # 3. 正则边界处理 (丢弃不完整语义)
        # 头部：若 ext_start > 0，向后找第一个标点作为新起点
        actual_start_offset = 0
        if ext_start > 0:
            match = SENTENCE_END_REGEX.search(segment)
            if match:
                actual_start_offset = match.end()
            # 如果没找到标点，保持从0开始（可能是一整句）

        # 尾部：若 ext_end < len(parent_text)，向前找最后一个标点作为终点
        actual_end_offset = len(segment)
        if ext_end < len(parent_text):
            matches = list(SENTENCE_END_REGEX.finditer(segment))
            if matches:
                actual_end_offset = matches[-1].end()

        final_segment = segment[actual_start_offset:actual_end_offset]

        # 4. 最终 Token 校验
        if TokenBudgetManager.count_tokens(final_segment) > budget:
            final_segment = TokenBudgetManager._truncate_text_smart(final_segment, budget)

        return final_segment

    @staticmethod
    def _truncate_text_smart(text: str, max_tokens: int) -> str:
        if not text: return ""
        tokens = ENCODER.encode(text)
        if len(tokens) <= max_tokens: return text
        if max_tokens < 50: return ENCODER.decode(tokens[:max_tokens])

        head_len = int(max_tokens * 0.3)
        tail_len = int(max_tokens * 0.2)
        head_text = ENCODER.decode(tokens[:head_len])
        tail_text = ENCODER.decode(tokens[-tail_len:])
        return f"{head_text}\n\n... [Omitted] ...\n\n{tail_text}"

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        if not text: return ""
        tokens = ENCODER.encode(text)
        if len(tokens) <= max_tokens: return text
        return ENCODER.decode(tokens[:max_tokens])

    @staticmethod
    def _compress_log_output(log_text: str, max_tokens: int) -> str:
        lines = log_text.strip().split("\n")
        if not lines: return ""

        error_keywords = ['ERROR', 'Exception', 'Fatal', 'Crash', 'Timeout', 'OOMKilled', 'CrashLoopBackOff']
        important_indices = set()
        error_types = {}

        for i, line in enumerate(lines):
            for kw in error_keywords:
                if kw in line:
                    error_types[kw] = error_types.get(kw, 0) + 1
            if any(kw in line for kw in error_keywords):
                for j in range(max(0, i - 2), min(len(lines), i + 3)):
                    important_indices.add(j)

        result_parts = []
        summary_header = f"[Log Summary: Total {len(lines)} lines, Key Lines: {len(important_indices)}]"
        if error_types:
            stats_str = json.dumps(error_types, ensure_ascii=False)
            if len(stats_str) > 200: stats_str = stats_str[:200] + "...}"
            summary_header += f" | Errors: {stats_str}"
        result_parts.append(summary_header)

        head_end = min(15, len(lines))
        result_parts.append("--- START ---")
        result_parts.extend(lines[:head_end])
        kept_indices = set(range(head_end))

        middle_lines = [lines[idx] for idx in sorted(important_indices) if idx not in kept_indices]
        if middle_lines:
            result_parts.append("--- KEY ERRORS ---")
            result_parts.extend(middle_lines)

        tail_start = max(0, len(lines) - 15)
        remaining_tail = set(range(tail_start, len(lines))) - kept_indices
        if remaining_tail:
            result_parts.append("--- END ---")
            result_parts.extend(lines[tail_start:])

        compressed = '\n'.join(result_parts)
        if TokenBudgetManager.count_tokens(compressed) > max_tokens:
            return TokenBudgetManager._truncate_text(compressed, max_tokens)
        return compressed