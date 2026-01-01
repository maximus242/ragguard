# Framework Integrations

RAGGuard integrates with popular AI/ML frameworks.

## LangChain

Native LangChain compatibility for chains and agents:

```python
from langchain.chains import RetrievalQA
from ragguard.integrations.langchain import LangChainSecureRetriever

retriever = LangChainSecureRetriever(
    qdrant_client=client,
    collection="documents",
    policy=policy,
    embedding_function=embeddings.embed_query
)

# Set user context
retriever.set_user({"id": "alice", "roles": ["engineer"]})

# Use in any LangChain chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa.run("What were our Q3 results?")
```

## LlamaIndex

Wrap any LlamaIndex retriever with RAGGuard permissions:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from ragguard.integrations.llamaindex import wrap_retriever
from ragguard import load_policy

# Your existing LlamaIndex setup
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
base_retriever = index.as_retriever()

# Wrap with RAGGuard
policy = load_policy("policy.yaml")
secure_retriever = wrap_retriever(base_retriever, policy)

# Query with permissions
results = secure_retriever.retrieve(
    "What is our company policy?",
    user_context={"department": "hr", "roles": ["manager"]}
)
```

### SecureQueryEngine

```python
from ragguard.integrations.llamaindex import SecureQueryEngine

query_engine = SecureQueryEngine(index=index, policy=policy)

response = query_engine.query(
    "What were our Q3 results?",
    user_context={"department": "finance", "roles": ["analyst"]}
)
```

## LangGraph

Use RAGGuard with LangGraph for stateful agents:

```python
from ragguard.integrations.langgraph import SecureRetrieverNode, create_secure_retriever_tool
from langgraph.graph import StateGraph

# Create secure retriever node for LangGraph
node = SecureRetrieverNode(
    qdrant_client=client,
    collection="documents",
    policy=policy,
    embedding_function=embeddings.embed_query
)

# Or create as a tool for agent use
tool = create_secure_retriever_tool(
    qdrant_client=client,
    collection="documents",
    policy=policy,
    embedding_function=embeddings.embed_query,
    name="secure_search",
    description="Search documents with access control"
)
```

## CrewAI

Integrate RAGGuard with CrewAI for multi-agent systems:

```python
from ragguard.integrations.crewai import SecureSearchTool
from crewai import Agent, Task, Crew

# Create secure search tool for CrewAI agents
search_tool = SecureSearchTool(
    retriever=retriever,  # Any RAGGuard retriever
    name="secure_document_search",
    description="Search documents with permission filtering"
)

# Use in CrewAI agent
researcher = Agent(
    role="Research Analyst",
    goal="Find relevant documents",
    tools=[search_tool],
    verbose=True
)

# CrewAI automatically passes user context
task = Task(
    description="Find Q3 revenue reports",
    agent=researcher,
    context={"user": {"id": "alice", "department": "finance"}}
)
```

## DSPy

```python
from ragguard.integrations.dspy import RAGGuardRM, SecureRetrieve

# Create DSPy retriever model
rm = RAGGuardRM(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    k=5
)

# Configure DSPy to use this retriever
dspy.settings.configure(rm=rm)
```

## AWS Bedrock

```python
from ragguard.integrations.aws_bedrock import BedrockSecureRetriever

retriever = BedrockSecureRetriever(
    knowledge_base_id="your-kb-id",
    policy=policy
)
```
