"""
LangGraph 集成 MCP Client，支持动态工具发现和管理
"""
import asyncio
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

from mcp_server import mcp_server, MCPToolResult

logger = logging.getLogger("MCPClient")


class MCPToolWrapper(BaseTool):
    """将 MCP 工具包装为 LangChain Tool"""
    name: str
    description: str
    mcp_tool_name: str
    input_schema: Dict[str, Any]
    
    async def _arun(self, **kwargs) -> str:
        """异步执行 MCP 工具"""
        result: MCPToolResult = await mcp_server.call_tool(
            tool_name=self.mcp_tool_name,
            arguments=kwargs
        )
        
        if result.success:
            # 将结果转换为字符串
            import json
            return json.dumps(result.data, indent=2, ensure_ascii=False)
        else:
            return f"Error: {result.error}"
    
    def _run(self, **kwargs) -> str:
        """同步执行（调用异步版本）"""
        return asyncio.run(self._arun(**kwargs))


class MCPClient:
    """MCP 客户端 - 工具发现和管理"""
    
    def __init__(self):
        self.available_tools: Dict[str, MCPToolWrapper] = {}
        logger.info("MCP Client initialized")
    
    async def discover_tools(self) -> List[MCPToolWrapper]:
        """发现所有可用的 MCP 工具"""
        tools_info = await mcp_server.list_tools()
        
        wrapped_tools = []
        for tool_info in tools_info:
            wrapper = MCPToolWrapper(
                name=tool_info["name"],
                description=tool_info["description"],
                mcp_tool_name=tool_info["name"],
                input_schema=tool_info["input_schema"]
            )
            self.available_tools[tool_info["name"]] = wrapper
            wrapped_tools.append(wrapper)
        
        logger.info(f"Discovered {len(wrapped_tools)} MCP tools")
        return wrapped_tools
    
    def get_tool(self, tool_name: str) -> Optional[MCPToolWrapper]:
        """获取指定工具"""
        return self.available_tools.get(tool_name)
    
    def get_all_tools(self) -> List[MCPToolWrapper]:
        """获取所有已发现的工具"""
        return list(self.available_tools.values())
    
    def filter_tools_by_prefix(self, prefix: str) -> List[MCPToolWrapper]:
        """按前缀过滤工具"""
        return [
            tool for name, tool in self.available_tools.items()
            if name.startswith(prefix)
        ]


# 全局 MCP Client 实例
mcp_client = MCPClient()


async def get_mcp_tools_for_agent(tool_category: Optional[str] = None) -> List[MCPToolWrapper]:
    """为 Agent 获取 MCP 工具列表
    
    Args:
        tool_category: 工具类别过滤，如 "monitor", "db"，None 表示全部
    
    Returns:
        LangChain Tool 列表
    """
    # 确保工具已发现
    if not mcp_client.available_tools:
        await mcp_client.discover_tools()
    
    if tool_category:
        return mcp_client.filter_tools_by_prefix(tool_category)
    else:
        return mcp_client.get_all_tools()
