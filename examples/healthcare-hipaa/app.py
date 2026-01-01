"""
HIPAA-Compliant Medical Records Q&A System

This application demonstrates how to build a production-ready healthcare
application with RAGGuard for permission-aware retrieval.

Features:
- Role-based access control (Patient, Doctor, Nurse, Admin)
- HIPAA-compliant audit logging
- Semantic search over medical records
- REST API with FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

# Import RAGGuard
import sys
sys.path.insert(0, '../../')  # Add parent directory to path
from ragguard import ChromaDBSecureRetriever, load_policy, AuditLogger

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare HIPAA Q&A System",
    description="Permission-aware medical records search with RAGGuard",
    version="1.0.0"
)

# Initialize embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="medical_records",
    metadata={"description": "HIPAA-compliant medical records"}
)

# Load policy
print("Loading access control policy...")
policy = load_policy("policy.yaml")

# Initialize audit logger (HIPAA compliance requirement)
print("Initializing audit logger...")
audit_logger = AuditLogger(output="file:./audit.jsonl")

# Create secure retriever
print("Creating secure retriever...")
retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    embed_fn=embedder.encode,
    audit_logger=audit_logger
)

print("✓ Application initialized successfully")


# Request/Response Models
class UserContext(BaseModel):
    """User context for permission checks"""
    id: str
    roles: List[str]
    patient_id: Optional[str] = None
    department: Optional[str] = None
    assigned_patients: Optional[List[str]] = None


class QueryRequest(BaseModel):
    """Query request with user context"""
    query: str
    user: UserContext
    limit: int = 5


class SearchResult(BaseModel):
    """Single search result"""
    text: str
    patient_id: str
    department: str
    record_type: str
    distance: float


class QueryResponse(BaseModel):
    """Query response with results and audit info"""
    query: str
    results: List[SearchResult]
    results_count: int
    user_id: str
    user_roles: List[str]
    timestamp: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Healthcare HIPAA Q&A System",
        "version": "1.0.0",
        "ragguard": "enabled"
    }


@app.get("/stats")
async def stats():
    """Get database statistics"""
    count = collection.count()
    return {
        "total_records": count,
        "collection": "medical_records",
        "vector_db": "ChromaDB",
        "security": "RAGGuard permission-aware retrieval"
    }


@app.post("/query", response_model=QueryResponse)
async def query_records(request: QueryRequest):
    """
    Query medical records with automatic permission filtering.

    This endpoint demonstrates HIPAA-compliant access control:
    - Patients see only their own records
    - Doctors see their assigned patients
    - Nurses see patients in their department
    - Admins see all records (with audit trail)

    All queries are logged to audit.jsonl for compliance.
    """
    try:
        # Convert user context to dict
        user_dict = request.user.dict()

        # Additional audit logging (custom info)
        audit_info = {
            "endpoint": "/query",
            "client_ip": "127.0.0.1",  # In production, get from request
            "timestamp": datetime.utcnow().isoformat()
        }

        # Perform secure search
        results = retriever.search(
            query=request.query,
            user=user_dict,
            limit=request.limit
        )

        # Format results
        formatted_results = []
        for result in results:
            # ChromaDB result format
            metadata = result.get('metadata', {})
            document = result.get('document', '')
            distance = result.get('distance', 0.0)

            formatted_results.append(SearchResult(
                text=document,
                patient_id=metadata.get('patient_id', 'unknown'),
                department=metadata.get('department', 'unknown'),
                record_type=metadata.get('record_type', 'general'),
                distance=distance
            ))

        # Return response
        return QueryResponse(
            query=request.query,
            results=formatted_results,
            results_count=len(formatted_results),
            user_id=request.user.id,
            user_roles=request.user.roles,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        # Log error
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/recent")
async def get_recent_audit_logs(limit: int = 10):
    """
    Get recent audit log entries.

    In production, this should be restricted to admin users only.
    """
    try:
        with open("audit.jsonl", "r") as f:
            lines = f.readlines()
            recent_lines = lines[-limit:]
            logs = [json.loads(line) for line in recent_lines]
            return {
                "total_logs": len(lines),
                "recent_logs": logs
            }
    except FileNotFoundError:
        return {
            "total_logs": 0,
            "recent_logs": [],
            "message": "No audit logs yet"
        }


@app.get("/users/sample")
async def get_sample_users():
    """
    Get sample user contexts for testing.

    In production, users would come from your authentication system.
    """
    return {
        "patient": {
            "id": "john.smith@email.com",
            "roles": ["patient"],
            "patient_id": "P001",
            "example_query": "What is my blood pressure?"
        },
        "doctor": {
            "id": "dr.williams@hospital.com",
            "roles": ["doctor"],
            "department": "cardiology",
            "assigned_patients": ["P001"],
            "example_query": "Show recent visits for my patients"
        },
        "nurse": {
            "id": "nurse.davis@hospital.com",
            "roles": ["nurse"],
            "department": "cardiology",
            "example_query": "Show cardiology patient medications"
        },
        "admin": {
            "id": "admin.brown@hospital.com",
            "roles": ["admin"],
            "example_query": "Show all recent admissions"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🏥 Healthcare HIPAA Q&A System")
    print("="*60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Sample users: http://localhost:8000/users/sample")
    print("Audit logs: http://localhost:8000/audit/recent")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
