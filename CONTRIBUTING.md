# Contributing to RAGGuard

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ragguard.git
   cd ragguard
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest
   ```

4. Run linting:
   ```bash
   ruff check .
   mypy ragguard
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add: brief description of change"
   ```

6. Push and open a pull request

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if needed
- Ensure all CI checks pass

## Code Style

- Follow existing code patterns
- Use type hints
- Keep functions focused and small

## Running Integration Tests

Integration tests require Docker:

```bash
docker-compose up -d
pytest -m integration
docker-compose down
```

## Questions?

Open an issue on GitHub.
