# AutoGen Integration

RAGGuard integrates with Microsoft's AutoGen framework for building multi-agent AI applications with permission-aware document retrieval.

## Installation

```bash
pip install ragguard autogen-agentchat autogen-ext[openai]
```

## Quick Start

```python
from ragguard.integrations.autogen import create_secure_search_tool
from ragguard import QdrantSecureRetriever, Policy

# Create policy
policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}
    ],
    "default": "deny"
})

# Create retriever
retriever = QdrantSecureRetriever(
    client=qdrant_client,
    collection="documents",
    policy=policy,
    embed_fn=embed_function
)

# Create secure search tool
search_tool = create_secure_search_tool(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"}
)

# Use with AutoGen agent
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4")
assistant = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[search_tool]
)
```

## Components

### SecureSearchTool

For AutoGen v0.4+:

```python
from ragguard.integrations.autogen import SecureSearchTool

tool = SecureSearchTool(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    max_results=10,
    name="search_documents",
    description="Search internal documents"
)

# Call directly
result = tool("machine learning papers")

# Or convert to FunctionTool
function_tool = tool.as_function_tool()
```

### SecureRetrieverFunction

For AutoGen v0.2 (legacy):

```python
from ragguard.integrations.autogen import SecureRetrieverFunction

search_fn = SecureRetrieverFunction(
    retriever=retriever,
    user={"id": "alice"},
    name="search_docs"
)

# Register with agent
from autogen import AssistantAgent

assistant = AssistantAgent(
    name="researcher",
    llm_config={
        "functions": [search_fn.function_schema],
        "config_list": config_list
    }
)
assistant.register_function(
    function_map={"search_docs": search_fn}
)
```

### SecureRAGAgent

Pre-configured agent with secure retrieval:

```python
from ragguard.integrations.autogen import SecureRAGAgent

rag_agent = SecureRAGAgent(
    name="researcher",
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    model="gpt-4",
    system_message="You are a research assistant."
)

# Update user context
rag_agent.set_user({"id": "bob", "department": "sales"})
```

## User Context

Always set user context before searches:

```python
# Set during creation
tool = SecureSearchTool(
    retriever=retriever,
    user={"id": "alice", "department": "engineering", "roles": ["analyst"]}
)

# Update later
tool.set_user({"id": "bob", "department": "sales"})
```

## Multi-Agent Workflows

```python
from autogen_agentchat.teams import RoundRobinGroupChat

# Create agents with different access levels
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    tools=[search_tool]  # Has engineering access
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    tools=[search_tool]  # Same tool, but can update user context
)

# Before reviewer uses tool, update context
search_tool.set_user({"id": "reviewer", "roles": ["reviewer"]})

# Create team
team = RoundRobinGroupChat([researcher, reviewer])
```

## Error Handling

```python
from ragguard.exceptions import RetrieverError

try:
    result = tool("search query")
except RetrieverError as e:
    print(f"Search failed: {e}")
```

## Best Practices

1. **Always set user context** before agent execution
2. **Use method chaining** for fluent API: `tool.set_user(user).search(query)`
3. **Prefer SecureSearchTool** for AutoGen v0.4+
4. **Configure max_results** to balance relevance and context window
