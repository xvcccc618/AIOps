import json
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from bone import OpsAgentState, AgentStatus
from rca_schema import RCAResult, ActionItemType
from rca_ingestion import knowledge_ingestion_node

logger = logging.getLogger("RCARenderer")


# ================= 1. Markdown 渲染器 =================

def render_rca_to_markdown(rca_data: Dict[str, Any]) -> str:
    """
    将 RCA JSON 数据转换为精美的 Markdown 格式
    """
    try:
        if isinstance(rca_data, dict):
            rca = RCAResult(**rca_data)
        else:
            rca = rca_data

        md_lines = []

        # Header
        md_lines.append("# 故障根因分析报告 (RCA)")
        md_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**置信度**: {rca.confidence_score:.2%}")
        md_lines.append("---")

        # 1. Incident Summary
        md_lines.append("## 1. 故障简述")
        md_lines.append(f"> {rca.incident_summary}")
        md_lines.append("")

        # 2. Timeline
        md_lines.append("## 2. 故障时间线")
        md_lines.append("| 时间 | 类型 | 描述 |")
        md_lines.append("| :--- | :--- | :--- |")
        for event in rca.timeline:
            md_lines.append(f"| {event.timestamp} | {event.event_type} | {event.description} |")
        md_lines.append("")

        # 3. Root Cause Analysis
        md_lines.append("## 3. 根因分析 (5 Whys)")
        md_lines.append(rca.root_cause_analysis)
        md_lines.append("")

        if rca.evidence_chain:
            md_lines.append("## 4. 证据链 (Evidence Chain)")
            for i, ev in enumerate(rca.evidence_chain):
                md_lines.append(f"**证据 {i + 1} [{ev.type}]**: `{ev.content}`")
                md_lines.append(f"- *相关性*: {ev.relevance}")
            md_lines.append("")

        # 5. Impact Assessment
        md_lines.append("## 5. 影响评估")
        md_lines.append(f"{rca.impact_assessment}")
        md_lines.append("")

        # 6. Action Items
        md_lines.append("## 6. 改进措施 (Action Items)")
        md_lines.append("| 类型 | 描述 | 负责人 | 截止日期 | 优先级 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for item in rca.action_items:
            type_cn = {
                ActionItemType.SHORT_TERM: "短期修复",
                ActionItemType.MEDIUM_TERM: "⚙中期优化",
                ActionItemType.LONG_TERM: "长期架构"
            }.get(item.type, item.type)
            md_lines.append(f"| {type_cn} | {item.description} | {item.owner} | {item.deadline} | {item.priority} |")
        md_lines.append("")

        # 7. Related Artifacts
        if rca.related_artifacts:
            md_lines.append("## 7. 关联资产")
            for art in rca.related_artifacts:
                md_lines.append(f"- **[{art.type}]** {art.name}: `{art.url_or_id}`")
            md_lines.append("")

        return "\n".join(md_lines)

    except Exception as e:
        logger.error(f"Error rendering markdown: {e}")
        return f"Error rendering report: {str(e)}"


# ================= 2. HTML 邮件渲染器 =================

def render_rca_to_html(rca_data: Dict[str, Any]) -> str:
    """
    生成简单的 HTML 邮件模板
    """
    try:
        if isinstance(rca_data, dict):
            rca = RCAResult(**rca_data)
        else:
            rca = rca_data

        def gen_table(headers, rows):
            th = "".join([f"<th>{h}</th>" for h in headers])
            trs = ""
            for row in rows:
                tds = "".join([f"<td>{cell}</td>" for cell in row])
                trs += f"<tr>{tds}</tr>"
            return f"<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>{th}{trs}</table>"

        timeline_rows = [[e.timestamp, e.event_type, e.description] for e in rca.timeline]
        action_rows = [
            [item.type.value, item.description, item.owner, item.deadline, item.priority]
            for item in rca.action_items
        ]

        evidence_rows = [[ev.type, ev.content[:100] + "...", ev.relevance] for ev in rca.evidence_chain]

        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h1 style="color: #d9534f;">故障根因分析报告 (RCA)</h1>
                <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">1. 故障简述</h2>
                <blockquote style="background: #f9f9f9; border-left: 5px solid #d9534f; padding: 10px; margin: 10px 0;">
                    {rca.incident_summary}
                </blockquote>

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">2. 故障时间线</h2>
                {gen_table(["时间", "类型", "描述"], timeline_rows)}

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">3. 根因分析</h2>
                <p>{rca.root_cause_analysis.replace(chr(10), '<br>')}</p>

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">4. 证据链</h2>
                {gen_table(["类型", "内容摘要", "相关性"], evidence_rows) if evidence_rows else "<p>无明确证据链</p>"}

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">5. 影响评估</h2>
                <p>{rca.impact_assessment}</p>

                <h2 style="border-bottom: 2px solid #eee; padding-bottom: 5px;">6. 改进措施</h2>
                {gen_table(["类型", "描述", "负责人", "截止日期", "优先级"], action_rows)}

                <hr>
                <p style="font-size: 12px; color: #777;">此报告由 AI 助手自动生成，请人工复核。</p>
            </div>
        </body>
        </html>
        """
        return html_content
    except Exception as e:
        logger.error(f"Error rendering HTML: {e}")
        return f"<html><body>Error generating report: {str(e)}</body></html>"


# ================= 3. 分发模拟服务 =================

async def simulate_feishu_webhook(markdown_content: str) -> Dict[str, Any]:
    """模拟发送飞书/企微机器人"""
    logger.info("[Distribution] Simulating Feishu/Webhook send...")
    await asyncio.sleep(0.5)
    return {
        "status": "success",
        "doc_url": "https://feishu.example.com/doc/RCA-2025-001",
        "message": "Feishu Doc created successfully"
    }


async def simulate_jira_api(rca_json: Dict[str, Any]) -> Dict[str, Any]:
    """模拟创建 Jira 工单"""
    logger.info("[Distribution] Simulating Jira API call...")
    await asyncio.sleep(0.5)
    return {
        "status": "success",
        "ticket_id": f"OPS-{datetime.now().year}-{1000 + int(datetime.now().timestamp()) % 1000}",
        "message": "Jira ticket created"
    }


async def distribute_and_ingest_rca(state: OpsAgentState, config: RunnableConfig) -> dict:
    """
    分发与反哺节点：
    1. 读取 rca_report_json
    2. 渲染 Markdown 和 HTML
    3. 模拟调用外部 API (Feishu, Jira)
    4. 触发知识反哺 (Ingestion)
    5. 更新 distribution_status
    """
    rca_json = state.get("rca_report_json")
    if not rca_json:
        return {"status": AgentStatus.FAILED, "error_log": [{"node": "distributor", "error": "No RCA report found"}]}

    try:
        # 1. 渲染
        md_content = render_rca_to_markdown(rca_json)
        html_content = render_rca_to_html(rca_json)

        # 2. 并行分发模拟
        feishu_res, jira_res = await asyncio.gather(
            simulate_feishu_webhook(md_content),
            simulate_jira_api(rca_json)
        )

        distribution_result = {
            "feishu": feishu_res,
            "jira": jira_res,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[Distribution] Completed. Feishu: {feishu_res['status']}, Jira: {jira_res['status']}")

        # 3. 触发知识反哺
        ingestion_result = await knowledge_ingestion_node(state, config)

        final_msg = f"RCA 报告已生成、分发并回流知识库。\n-  飞书文档: {feishu_res.get('doc_url')}\n-  Jira 工单: {jira_res.get('ticket_id')}\n-  知识库更新: {'成功' if ingestion_result['status'] == AgentStatus.SUCCESS else '失败'}"

        return {
            "rca_distribution_status": "COMPLETED",
            "rca_distribution_details": distribution_result,
            "rca_ingestion_status": ingestion_result.get("rca_ingestion_status"),
            "final_answer": final_msg,
            "status": AgentStatus.SUCCESS
        }

    except Exception as e:
        logger.error(f"[Distribution] Failed: {e}")
        return {
            "rca_distribution_status": "FAILED",
            "status": AgentStatus.FAILED,
            "error_log": [{"node": "distributor", "error": str(e)}]
        }