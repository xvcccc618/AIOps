from typing import Dict, List, Set, Optional
from enum import Enum
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("RBAC")


class Role(str, Enum):
    """用户角色定义"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Permission(str, Enum):
    """权限级别定义"""
    READ = "read"           # 只读操作
    EXECUTE = "execute"     # 执行安全工具
    DANGEROUS = "dangerous" # 执行危险工具
    ADMIN = "admin"         # 管理员权限


class RBACPolicy(BaseModel):
    """RBAC 权限策略"""
    role_permissions: Dict[Role, Set[Permission]] = Field(default_factory=dict)
    tool_required_permissions: Dict[str, Permission] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class RBACManager:
    """RBAC 权限管理器"""
    
    def __init__(self):
        self.policy = self._create_default_policy()
        logger.info("RBAC Manager initialized with default policy")
    
    def _create_default_policy(self) -> RBACPolicy:
        """创建默认权限策略"""
        role_permissions = {
            Role.ADMIN: {Permission.READ, Permission.EXECUTE, Permission.DANGEROUS, Permission.ADMIN},
            Role.OPERATOR: {Permission.READ, Permission.EXECUTE},
            Role.VIEWER: {Permission.READ}
        }
        
        # 工具权限映射
        tool_required_permissions = {
            # 只读工具
            "query_prometheus_metrics": Permission.READ,
            "fetch_k8s_pod_logs": Permission.READ,
            "get_k8s_pod_status": Permission.READ,
            "query_service_dependencies": Permission.READ,
            "analyze_slow_queries": Permission.READ,
            "execute_readonly_query": Permission.READ,
            "check_database_health": Permission.READ,
            "analyze_lock_contention": Permission.READ,
            
            # 执行工具
            "restart_service": Permission.DANGEROUS,
            "scale_deployment": Permission.DANGEROUS,
            "kill_pod": Permission.DANGEROUS,
            "execute_query": Permission.EXECUTE,
        }
        
        return RBACPolicy(
            role_permissions=role_permissions,
            tool_required_permissions=tool_required_permissions
        )
    
    def check_permission(self, user_role: str, tool_name: str) -> bool:
        """检查用户是否有权限执行工具
        
        Args:
            user_role: 用户角色 (admin/operator/viewer)
            tool_name: 工具名称
        
        Returns:
            True 表示有权限，False 表示无权限
        """
        try:
            role = Role(user_role)
        except ValueError:
            logger.warning(f"Unknown role: {user_role}, treating as viewer")
            role = Role.VIEWER
        
        # 获取工具所需权限
        required_permission = self.policy.tool_required_permissions.get(
            tool_name, Permission.READ
        )
        
        # 检查用户角色是否有该权限
        user_permissions = self.policy.role_permissions.get(role, set())
        has_permission = required_permission in user_permissions
        
        if not has_permission:
            logger.warning(
                f"Permission denied: role={role.value}, tool={tool_name}, "
                f"required={required_permission.value}"
            )
        
        return has_permission
    
    def filter_tools_by_role(self, tool_names: List[str], user_role: str) -> List[str]:
        """根据用户角色过滤可用工具
        
        Args:
            tool_names: 工具名称列表
            user_role: 用户角色
        
        Returns:
            用户有权使用的工具列表
        """
        return [
            tool for tool in tool_names
            if self.check_permission(user_role, tool)
        ]
    
    def get_user_permissions(self, user_role: str) -> Set[str]:
        """获取用户角色的所有权限
        
        Args:
            user_role: 用户角色
        
        Returns:
            权限名称集合
        """
        try:
            role = Role(user_role)
        except ValueError:
            role = Role.VIEWER
        
        permissions = self.policy.role_permissions.get(role, set())
        return {p.value for p in permissions}
    
    def add_tool_permission(self, tool_name: str, required_permission: Permission):
        """添加工具权限要求
        
        Args:
            tool_name: 工具名称
            required_permission: 所需权限
        """
        self.policy.tool_required_permissions[tool_name] = required_permission
        logger.info(f"Added permission requirement: {tool_name} -> {required_permission.value}")
    
    def is_dangerous_tool(self, tool_name: str) -> bool:
        """判断工具是否为危险工具
        
        Args:
            tool_name: 工具名称
        
        Returns:
            True 表示是危险工具
        """
        required = self.policy.tool_required_permissions.get(tool_name, Permission.READ)
        return required == Permission.DANGEROUS


# 全局 RBAC 管理器实例
rbac_manager = RBACManager()


def check_tool_permission(user_role: str, tool_name: str) -> bool:
    """便捷函数：检查工具权限"""
    return rbac_manager.check_permission(user_role, tool_name)


def filter_available_tools(tool_names: List[str], user_role: str) -> List[str]:
    """便捷函数：过滤可用工具"""
    return rbac_manager.filter_tools_by_role(tool_names, user_role)
