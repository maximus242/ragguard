'use client'

import { useState } from 'react'

interface SearchResult {
  id: string
  score: number
  payload: Record<string, any>
}

interface SearchResponse {
  success: boolean
  results: SearchResult[]
  count: number
  error?: string
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Mock user context - in production, get from auth session
  const currentUser = {
    id: 'user-123',
    role: 'employee',
    department: 'engineering'
  }

  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          user: currentUser,
          limit: 10
        })
      })

      const data: SearchResponse = await response.json()

      if (!data.success) {
        throw new Error(data.error || 'Search failed')
      }

      setResults(data.results)
    } catch (err) {
      console.error('Search error:', err)
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      handleSearch()
    }
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <h1>RAGGuard Search</h1>
      <p style={{ color: '#666', marginBottom: '2rem' }}>
        Permission-aware document search with RAGGuard
      </p>

      <div style={{ marginBottom: '1rem' }}>
        <strong>Current User:</strong> {currentUser.id}
        ({currentUser.role}, {currentUser.department})
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search documents..."
          disabled={loading}
          style={{
            flex: 1,
            padding: '0.75rem',
            fontSize: '1rem',
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: loading ? '#ccc' : '#0070f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && (
        <div style={{
          padding: '1rem',
          marginBottom: '1rem',
          backgroundColor: '#fee',
          border: '1px solid #fcc',
          borderRadius: '4px',
          color: '#c00'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {results.length > 0 && (
        <div>
          <h2>Results ({results.length})</h2>
          {results.map((result) => (
            <div
              key={result.id}
              style={{
                padding: '1rem',
                marginBottom: '1rem',
                border: '1px solid #ddd',
                borderRadius: '4px',
                backgroundColor: '#fafafa'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <strong>ID: {result.id}</strong>
                <span style={{ color: '#0070f3' }}>
                  Score: {result.score.toFixed(4)}
                </span>
              </div>
              <div style={{
                backgroundColor: '#fff',
                padding: '0.75rem',
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                overflow: 'auto'
              }}>
                <pre style={{ margin: 0 }}>
                  {JSON.stringify(result.payload, null, 2)}
                </pre>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && results.length === 0 && query && !error && (
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          color: '#666'
        }}>
          No results found for "{query}"
        </div>
      )}
    </div>
  )
}
