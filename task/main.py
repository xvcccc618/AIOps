"""
IT-Ops Agent 主入口
企业级智能 IT 运维与工单排查 Agent
"""
import asyncio
import logging
import os
from typing import Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from bone import OpsAgentState, AgentStatus, IncidentSeverity, parse_incident_severity
from route import build_graph
from multi_agent_graph import build_multi_agent_graph
from evaluation import evaluation_system
from hallucination_guard import timeout_manager
from tool import ALL_TOOLS

# 结构化日志：structlog 桥接 stdlib logging，各模块的 logging 调用无需改动
from observability import setup_logging, LLMetricsHandler
from settings import get_llm_config

setup_logging()
logger = logging.getLogger("Main")


def create_llm_instance(api_key: str = None, model: str = None) -> ChatOpenAI:
    """创建 LLM 实例（OpenAI 兼容接口；base_url / model 默认取自 .env）"""
    llm_cfg = get_llm_config()
    api_key = api_key or llm_cfg["api_key"]
    if not api_key:
        raise ValueError("请在 .env 中设置 OPENAI_API_KEY")

    kwargs = dict(model=model or llm_cfg["model"], temperature=0.3, api_key=api_key)
    if llm_cfg["base_url"]:
        kwargs["base_url"] = llm_cfg["base_url"]
    return ChatOpenAI(**kwargs)


async def run_single_agent(
    query: str,
    severity: str = "P2",
    user_role: str = "standard",
    llm: ChatOpenAI = None
) -> Dict[str, Any]:
    """运行单 Agent 模式"""
    logger.info(f"Starting single agent for query: {query[:50]}...")
    
    # 构建图
    graph = await build_graph()
    
    # 初始状态
    initial_state: OpsAgentState = {
        "query": query,
        "incident_severity": parse_incident_severity(severity),
        "user_permission_level": user_role,
        "messages": [],
        "execution_history": [],
        "failed_paths": [],
        "system_hints": [],
        "error_log": [],
        "available_tools": ALL_TOOLS,
    }
    
    # 配置（注入 LLM 指标采集：调用次数 / token / 耗时）
    llm_metrics = LLMetricsHandler()
    config = {
        "configurable": {
            "llm_instance": llm,
            "thread_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "callbacks": [llm_metrics]
    }
    
    # 运行
    start_time = datetime.now()
    session_id = config["configurable"]["thread_id"]
    
    try:
        result = await graph.ainvoke(initial_state, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        # 追踪指标
        final_confidence = 0.0
        if "rca_result" in result:
            final_confidence = result["rca_result"].get("confidence_score", 0.0)
        
        evaluation_system.track_agent_behavior(
            session_id=session_id,
            execution_history=result.get("execution_history", []),
            total_duration=duration,
            final_confidence=final_confidence,
            total_tokens=llm_metrics.summary()["total_tokens"]
        )
        
        return {
            "status": "SUCCESS",
            "session_id": session_id,
            "duration_seconds": duration,
            "final_answer": result.get("final_answer", ""),
            "rca_result": result.get("rca_result"),
            "execution_history": result.get("execution_history", [])
        }
    
    except Exception as e:
        logger.error(f"Single agent execution failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "session_id": session_id
        }


async def run_multi_agent(
    query: str,
    severity: str = "P2",
    llm: ChatOpenAI = None
) -> Dict[str, Any]:
    """运行多 Agent 协作模式"""
    logger.info(f"Starting multi-agent collaboration for query: {query[:50]}...")
    
    # 构建多 Agent 图
    graph = await build_multi_agent_graph()
    
    # 初始状态
    initial_state = {
        "query": query,
        "incident_severity": parse_incident_severity(severity),
        "messages": [],
        "handoff_count": 0,
        "handoff_history": [],
        "specialist_findings": {},
        "is_arbitrating": False,
        "extracted_entities": {},
        "topology_context": ""
    }
    
    # 配置
    config = {
        "configurable": {
            "llm_instance": llm,
            "thread_id": f"multi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    }
    
    # 运行
    start_time = datetime.now()
    session_id = config["configurable"]["thread_id"]
    
    try:
        result = await graph.ainvoke(initial_state, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "SUCCESS",
            "session_id": session_id,
            "duration_seconds": duration,
            "final_answer": result.get("final_answer", ""),
            "specialist_findings": result.get("specialist_findings", {}),
            "handoff_count": result.get("handoff_count", 0)
        }
    
    except Exception as e:
        logger.error(f"Multi-agent execution failed: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "session_id": session_id
        }


async def interactive_mode(llm: ChatOpenAI):
    """交互式模式"""
    print("\n" + "="*60)
    print("IT-Ops Agent 交互式诊断系统")
    print("="*60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'mode' 切换单 Agent / 多 Agent 模式")
    print("输入 'report' 生成评测报告")
    print("="*60 + "\n")
    
    mode = "single"  # 默认单 Agent 模式
    
    while True:
        try:
            query = input("请输入故障描述: ").strip()
            
            if query.lower() in ["quit", "exit"]:
                print("再见！")
                break
            
            if query.lower() == "mode":
                mode = "multi" if mode == "single" else "single"
                print(f"已切换到 {mode} Agent 模式")
                continue
            
            if query.lower() == "report":
                report = evaluation_system.export_report()
                print("\n" + report + "\n")
                continue
            
            if not query:
                continue
            
            # 选择严重等级
            severity = input("严重等级 (P0/P1/P2/P3/P4) [P2]: ").strip() or "P2"
            
            # 运行
            print(f"\n正在 {mode} 模式下分析...\n")

            if mode == "single":
                result = await run_single_agent(query, severity, llm=llm)
            else:
                result = await run_multi_agent(query, severity, llm=llm)
            
            # 输出结果
            print("\n" + "="*60)
            print("分析结果")
            print("="*60)
            print(f"状态: {result['status']}")
            print(f"会话 ID: {result['session_id']}")
            print(f"耗时: {result['duration_seconds']:.2f} 秒")
            print("\n最终答案:")
            print(result.get("final_answer", "无"))
            
            if "rca_result" in result and result["rca_result"]:
                print("\nRCA 报告:")
                rca = result["rca_result"]
                print(f"  根因: {rca.get('root_cause_analysis', 'N/A')}")
                print(f"  置信度: {rca.get('confidence_score', 0):.2f}")
            
            print("="*60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n程序已中断，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IT-Ops Agent")
    parser.add_argument("--api-key", type=str, help="OpenAI API Key")
    parser.add_argument("--model", type=str, default=None, help="LLM 模型（默认取 .env 的 LLM_MODEL）")
    parser.add_argument("--query", type=str, help="直接运行查询（非交互模式）")
    parser.add_argument("--multi", action="store_true", help="使用多 Agent 模式")
    
    args = parser.parse_args()
    
    # 创建 LLM
    try:
        llm = create_llm_instance(api_key=args.api_key, model=args.model)
        logger.info(f"LLM initialized: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        print("无法初始化 LLM，请检查 API Key")
        return
    
    if args.query:
        # 直接运行模式
        if args.multi:
            result = await run_multi_agent(args.query, llm=llm)
        else:
            result = await run_single_agent(args.query, llm=llm)
        
        print(result)
    else:
        # 交互模式
        await interactive_mode(llm)


if __name__ == "__main__":
    asyncio.run(main())
