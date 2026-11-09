"""Utility modules for the SimAgent workflow

This package contains shared utilities used across all agents and workflow components:
- conversation_logger: Logging conversations and workflow state
- workflow_visualizer: Visualization of workflow graph and state
- log_utils: Logging utilities and helpers
- file_manager: Secure file operations with automatic registry tracking
- path_utils: Path sanitization and resolution utilities
"""

from .conversation_logger import (
    get_conversation_logger,
    set_log_file,
    log_user_prompt,
    log_supervisor_routing,
    log_agent_start,
    log_agent_action,
    log_file_operation,
    log_human_checkpoint,
    log_agent_completion,
    log_error,
    log_llm_interaction,
    log_workflow_completion,
)
from .log_utils import reconstruct_assistant_text
from .workflow_visualizer import WorkflowVisualizer
from .dynamic_tool_loader import DynamicToolLoader, get_dynamic_tool_loader
from .file_manager import SecureFileManager
from .path_utils import (
    sanitize_tool_output_params,
    normalize_to_filename,
    normalize_tool_params_for_agent,
    validate_output_in_agent_directory,
    ensure_agent_directory_exists,
    get_output_path_for_agent,
    resolve_input_file_path,
    resolve_tool_input_paths,
    get_path_resolution_code_snippet,
    register_file_in_registry
)

__all__ = [
    "get_conversation_logger",
    "set_log_file",
    "log_user_prompt",
    "log_supervisor_routing",
    "log_agent_start",
    "log_agent_action",
    "log_file_operation",
    "log_human_checkpoint",
    "log_agent_completion",
    "log_error",
    "log_llm_interaction",
    "log_workflow_completion",
    "reconstruct_assistant_text",
    "WorkflowVisualizer",
    "DynamicToolLoader",
    "get_dynamic_tool_loader",
    "SecureFileManager",
    "sanitize_tool_output_params",
    "normalize_to_filename",
    "normalize_tool_params_for_agent",
    "validate_output_in_agent_directory",
    "ensure_agent_directory_exists",
    "get_output_path_for_agent",
    "resolve_input_file_path",
    "resolve_tool_input_paths",
    "get_path_resolution_code_snippet",
    "register_file_in_registry",
]
