-- Initialize pgvector extension and create sample schema
-- This script runs automatically when the PostgreSQL container starts

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a sample documents table with vector embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(384),  -- 384-dimensional embeddings (adjust based on your model)

    -- Metadata for permission filtering
    visibility TEXT,
    department TEXT,
    confidential BOOLEAN DEFAULT false,
    owner_id TEXT,
    shared_with TEXT[],  -- Array of user IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Additional metadata
    metadata JSONB
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create indexes for common filter fields
CREATE INDEX IF NOT EXISTS documents_visibility_idx ON documents(visibility);
CREATE INDEX IF NOT EXISTS documents_department_idx ON documents(department);
CREATE INDEX IF NOT EXISTS documents_owner_id_idx ON documents(owner_id);

-- Insert some sample documents
INSERT INTO documents (text, embedding, visibility, department, confidential, owner_id)
VALUES
    ('Public engineering documentation',
     -- Random 384-dim vector (in practice, use real embeddings)
     array_fill(0.1, ARRAY[384])::vector,
     'public', 'engineering', false, 'alice'),

    ('Internal sales report',
     array_fill(0.2, ARRAY[384])::vector,
     'internal', 'sales', true, 'bob'),

    ('HR policy documents',
     array_fill(0.3, ARRAY[384])::vector,
     'internal', 'hr', false, 'carol')
ON CONFLICT DO NOTHING;

-- Grant permissions to ragguard user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ragguard;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ragguard;
