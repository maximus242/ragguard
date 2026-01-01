"""
Secure retrievers for different database backends.

Includes both vector databases and graph databases with permission-aware querying.
"""

# Vector database retrievers
from .arangodb import ArangoDBSecureRetriever
from .azure_search import AzureCognitiveSearchSecureRetriever, AzureSearchSecureRetriever
from .base import BaseSecureRetriever
from .chromadb import ChromaDBSecureRetriever
from .elasticsearch import ElasticsearchSecureRetriever, OpenSearchSecureRetriever
from .faiss import FAISSSecureRetriever

# Graph database retrievers
from .graph_base import BaseGraphRetriever
from .milvus import MilvusSecureRetriever, ZillizSecureRetriever
from .neo4j import Neo4jSecureRetriever
from .neptune import NeptuneSecureRetriever
from .pgvector import PgvectorSecureRetriever
from .pinecone import PineconeSecureRetriever
from .qdrant import QdrantSecureRetriever
from .tigergraph import TigerGraphSecureRetriever
from .weaviate import WeaviateSecureRetriever

__all__ = [
    # Base classes
    "BaseSecureRetriever",
    "BaseGraphRetriever",
    # Vector databases
    "QdrantSecureRetriever",
    "PgvectorSecureRetriever",
    "WeaviateSecureRetriever",
    "PineconeSecureRetriever",
    "ChromaDBSecureRetriever",
    "FAISSSecureRetriever",
    "MilvusSecureRetriever",
    "ZillizSecureRetriever",
    "ElasticsearchSecureRetriever",
    "OpenSearchSecureRetriever",
    "AzureSearchSecureRetriever",
    "AzureCognitiveSearchSecureRetriever",
    # Graph databases
    "Neo4jSecureRetriever",
    "NeptuneSecureRetriever",
    "TigerGraphSecureRetriever",
    "ArangoDBSecureRetriever",
]
