"""
实现 Model Context Protocol 服务器，支持工具注册、发现和执行
"""
import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("MCPServer")


class MCPTool(BaseModel):
    """MCP 工具定义"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    input_schema: Dict[str, Any] = Field(description="JSON Schema 格式的参数定义")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出格式定义")
    requires_auth: bool = Field(False, description="是否需要认证")
    timeout_seconds: int = Field(30, description="执行超时时间")


class MCPToolResult(BaseModel):
    """MCP 工具执行结果"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class MCPServer:
    """MCP 服务器实现"""
    
    def __init__(self, server_name: str = "ops-agent-mcp", version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        logger.info(f"MCP Server '{server_name}' v{version} initialized")
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
        output_schema: Optional[Dict[str, Any]] = None,
        requires_auth: bool = False,
        timeout_seconds: int = 30
    ) -> None:
        """注册工具"""
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            requires_auth=requires_auth,
            timeout_seconds=timeout_seconds
        )
        self.tools[name] = tool
        self.tool_handlers[name] = handler
        logger.info(f"Registered MCP tool: {name}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [tool.model_dump() for tool in self.tools.values()]
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        auth_token: Optional[str] = None
    ) -> MCPToolResult:
        """调用工具"""
        if tool_name not in self.tools:
            return MCPToolResult(success=False, error=f"Tool '{tool_name}' not found")
        
        tool = self.tools[tool_name]
        
        # 认证检查
        if tool.requires_auth and not auth_token:
            return MCPToolResult(success=False, error="Authentication required")
        
        # 执行工具
        handler = self.tool_handlers[tool_name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 异步执行并应用超时
            result = await asyncio.wait_for(
                handler(arguments),
                timeout=tool.timeout_seconds
            )
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return MCPToolResult(
                success=True,
                data=result,
                execution_time_ms=execution_time
            )
        except asyncio.TimeoutError:
            return MCPToolResult(
                success=False,
                error=f"Tool execution timeout after {tool.timeout_seconds}s"
            )
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return MCPToolResult(success=False, error=str(e))
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        
        if method == "tools/list":
            tools = await self.list_tools()
            return {"result": {"tools": tools}}
        
        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name")
            arguments = request.get("params", {}).get("arguments", {})
            auth_token = request.get("auth_token")
            
            result = await self.call_tool(tool_name, arguments, auth_token)
            return {"result": result.model_dump()}
        
        else:
            return {"error": f"Unknown method: {method}"}


# 全局 MCP Server 实例
mcp_server = MCPServer()
