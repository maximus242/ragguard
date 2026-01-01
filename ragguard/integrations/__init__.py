"""
Integrations with popular frameworks and permission services.

RAGGuard provides integrations for:

1. **Cloud Platforms**:
   - AWS Bedrock Knowledge Bases - Secure retrieval with policy enforcement

2. **Identity Providers (IAM)**:
   - Google Workspace (Directory, Groups, Drive)

3. **Framework Integrations**:
   - LangChain - Compatible retrievers
   - LangGraph - Secure retriever nodes and tools for agent workflows
   - LlamaIndex - Retriever wrappers and query engines
   - MCP (Model Context Protocol) - Expose retrieval as MCP tools/resources
   - A2A (Agent2Agent Protocol) - Expose retrieval as A2A agent skills
   - CrewAI - Secure retriever tools for multi-agent workflows
   - OpenAI Assistants - Secure retrieval for OpenAI Assistants API
   - AutoGen - Microsoft's multi-agent framework integration
   - DSPy - Stanford NLP's programming framework for LMs

Additional integrations available in ragguard-enterprise package:
Auth0, Okta, Microsoft Graph, OPA, Cerbos, OpenFGA, Permit.io
"""

import logging

logger = logging.getLogger(__name__)

__all__ = []


def _is_missing_dependency(error: ImportError) -> bool:
    """Check if ImportError is due to missing optional dependency."""
    msg = str(error).lower()
    # Common patterns for missing dependencies
    return (
        "no module named" in msg
        or "cannot import name" in msg
        or ("module" in msg and "not found" in msg)
    )

# LangChain integration
try:
    from .langchain import LangChainPgvectorSecureRetriever, LangChainSecureRetriever
    __all__.extend(["LangChainPgvectorSecureRetriever", "LangChainSecureRetriever"])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import LangChain integration: %s", e)

# LangGraph integration
try:
    from .langgraph import (
        RetrieverState,
        SecureRetrieverNode,
        SecureRetrieverTool,
        create_secure_retriever_node,
        create_secure_retriever_tool,
    )
    __all__.extend([
        "RetrieverState",
        "SecureRetrieverNode",
        "SecureRetrieverTool",
        "create_secure_retriever_node",
        "create_secure_retriever_tool"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import LangGraph integration: %s", e)

# LlamaIndex integration
try:
    from .llamaindex import SecureLlamaIndexRetriever, SecureQueryEngine, wrap_retriever
    __all__.extend([
        "SecureLlamaIndexRetriever",
        "SecureQueryEngine",
        "wrap_retriever"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import LlamaIndex integration: %s", e)

# Google Workspace integration moved to ragguard-enterprise
# Deprecated: Import from ragguard_enterprise.integrations instead
# try:
#     from .google_workspace import (...)
# except ImportError:
#     pass

# AWS Bedrock integration
try:
    from .aws_bedrock import BedrockKnowledgeBaseFilterBuilder, BedrockKnowledgeBaseSecureRetriever
    __all__.extend([
        "BedrockKnowledgeBaseFilterBuilder",
        "BedrockKnowledgeBaseSecureRetriever"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import AWS Bedrock integration: %s", e)

# MCP (Model Context Protocol) integration
try:
    from .mcp import MCPRetrieverTool, RAGGuardMCPServer, create_mcp_server
    __all__.extend([
        "MCPRetrieverTool",
        "RAGGuardMCPServer",
        "create_mcp_server"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import MCP integration: %s", e)

# CrewAI integration
try:
    from .crewai import SecureRAGTool, SecureSearchInput
    from .crewai import SecureRetrieverTool as CrewAISecureRetrieverTool
    from .crewai import create_secure_retriever_tool as create_crewai_retriever_tool
    __all__.extend([
        "CrewAISecureRetrieverTool",
        "SecureRAGTool",
        "SecureSearchInput",
        "create_crewai_retriever_tool"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import CrewAI integration: %s", e)

# A2A (Agent2Agent Protocol) integration
try:
    from .a2a import (
        RAGGuardA2AServer,
        RAGGuardAgentExecutor,
        create_a2a_executor,
        create_a2a_server,
    )
    __all__.extend([
        "RAGGuardA2AServer",
        "RAGGuardAgentExecutor",
        "create_a2a_server",
        "create_a2a_executor"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import A2A integration: %s", e)

# OpenAI Assistants integration
try:
    from .openai_assistants import (
        AssistantRunHandler,
        SecureAssistantRetriever,
        create_secure_function_tool,
    )
    __all__.extend([
        "SecureAssistantRetriever",
        "AssistantRunHandler",
        "create_secure_function_tool"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import OpenAI Assistants integration: %s", e)

# AutoGen integration
try:
    from .autogen import (
        SecureRAGAgent,
        SecureRetrieverFunction,
        SecureSearchTool,
        create_secure_search_tool,
    )
    __all__.extend([
        "SecureSearchTool",
        "SecureRetrieverFunction",
        "SecureRAGAgent",
        "create_secure_search_tool"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import AutoGen integration: %s", e)

# DSPy integration
try:
    from .dspy import (
        RAGGuardRM,
        SecureRAG,
        SecureRetrieve,
        configure_ragguard_rm,
    )
    __all__.extend([
        "RAGGuardRM",
        "SecureRetrieve",
        "SecureRAG",
        "configure_ragguard_rm"
    ])
except ImportError as e:
    if not _is_missing_dependency(e):
        logger.warning("Failed to import DSPy integration: %s", e)
