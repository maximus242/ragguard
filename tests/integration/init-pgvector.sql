-- Initialize pgvector extension for RAGGuard integration tests
-- This script runs automatically when the PostgreSQL container starts

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE ragguard_test TO ragguard;

-- Create a test schema (optional, for organization)
CREATE SCHEMA IF NOT EXISTS test_schema;
GRANT ALL ON SCHEMA test_schema TO ragguard;

-- Set default tablespace permissions
ALTER DATABASE ragguard_test SET search_path TO public, test_schema;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension initialized successfully for RAGGuard tests';
END $$;
