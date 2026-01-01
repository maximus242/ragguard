# DSPy Integration

RAGGuard integrates with Stanford NLP's DSPy framework for building LM programs with permission-aware retrieval.

## Installation

```bash
pip install ragguard dspy-ai
```

## Quick Start

```python
import dspy
from ragguard.integrations.dspy import configure_ragguard_rm
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

# Configure DSPy with RAGGuard
configure_ragguard_rm(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    k=5
)

# Now dspy.Retrieve() uses RAGGuard automatically
retrieve = dspy.Retrieve(k=3)
results = retrieve("machine learning papers")
print(results.passages)
```

## Components

### RAGGuardRM

Retriever Model for DSPy's global configuration:

```python
from ragguard.integrations.dspy import RAGGuardRM

rm = RAGGuardRM(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    k=5,
    text_field="content"  # Field containing document text
)

# Configure DSPy
dspy.settings.configure(rm=rm)

# Use in DSPy programs
retrieve = dspy.Retrieve(k=3)
passages = retrieve("query").passages
```

### SecureRetrieve

Module for direct use in DSPy programs:

```python
from ragguard.integrations.dspy import SecureRetrieve

class SecureRAGProgram(dspy.Module):
    def __init__(self, retriever, user):
        self.retrieve = SecureRetrieve(
            retriever=retriever,
            user=user,
            k=5
        )
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Use with different users
program = SecureRAGProgram(retriever, user={"id": "alice", "department": "eng"})
result = program("What are our sales numbers?")

# Switch user
program.retrieve.set_user({"id": "bob", "department": "sales"})
result = program("What are our sales numbers?")
```

### SecureRAG

Pre-built RAG module:

```python
from ragguard.integrations.dspy import SecureRAG

rag = SecureRAG(
    retriever=retriever,
    user={"id": "alice", "department": "engineering"},
    k=5
)

# Query
result = rag("What were our Q3 results?")
print(result.answer)
print(result.context)  # Retrieved passages

# Change user
rag.set_user({"id": "bob", "department": "finance"})
result = rag("What were our Q3 results?")  # Different results
```

## Building RAG Programs

### Basic RAG

```python
class BasicRAG(dspy.Module):
    def __init__(self, retriever, user):
        self.retrieve = SecureRetrieve(retriever=retriever, user=user, k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        context = "\n\n".join(passages)
        return self.generate(context=context, question=question)
```

### Multi-Hop RAG

```python
class MultiHopRAG(dspy.Module):
    def __init__(self, retriever, user):
        self.retrieve = SecureRetrieve(retriever=retriever, user=user, k=3)
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # First retrieval
        passages = self.retrieve(question).passages
        context = "\n\n".join(passages)

        # Generate follow-up query
        followup = self.generate_query(context=context, question=question)

        # Second retrieval
        more_passages = self.retrieve(followup.search_query).passages
        full_context = context + "\n\n" + "\n\n".join(more_passages)

        return self.generate_answer(context=full_context, question=question)
```

## User Context Management

```python
# Set during creation
rm = RAGGuardRM(retriever=retriever, user={"id": "alice"})

# Update later
rm.set_user({"id": "bob", "department": "sales"})

# Access current user
print(rm.user)
```

## Optimizing with DSPy

RAGGuard works with DSPy's optimization:

```python
from dspy.teleprompt import BootstrapFewShot

# Create trainset
trainset = [
    dspy.Example(question="What is our policy?", answer="..."),
    # ...
]

# Create program with secure retrieval
rag = SecureRAG(retriever=retriever, user=user)

# Optimize
optimizer = BootstrapFewShot(metric=my_metric)
optimized_rag = optimizer.compile(rag, trainset=trainset)
```

## Best Practices

1. **Configure RM globally** for simple programs
2. **Use SecureRetrieve** for fine-grained control
3. **Update user context** before each user's request
4. **Set appropriate k** to balance context and relevance
5. **Handle empty results** gracefully in your programs
