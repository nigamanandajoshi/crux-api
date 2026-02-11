# Contributing to Doc-Squeeze

Thank you for considering contributing to Doc-Squeeze! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- pip and virtualenv

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ClawSearch.git
   cd ClawSearch
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY if needed
   ```

5. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

6. **Run the development server:**
   ```bash
   python main.py
   ```

## Code Style

We use the following tools to maintain code quality:

- **Ruff** - Fast Python linter
- **Black** - Code formatter
- **mypy** - Static type checker
- **isort** - Import sorting

### Running Linters

```bash
# Run all checks
ruff check .
mypy main.py
black --check .

# Auto-fix issues
ruff check . --fix
black .
```

## Testing

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=main --cov-report=html

# Run integration tests
python test_agent.py
```

### Writing Tests

- Place unit tests in `tests/test_*.py`
- Use pytest fixtures for common setup
- Mock external API calls (Jina, Groq)
- Aim for >80% code coverage

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code passes all linters and type checks
   - All tests pass
   - New features have tests
   - Documentation is updated

3. **Commit your changes** with clear messages:
   ```bash
   git commit -m "feat: add new feature X"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

4. **Push to your fork** and submit a pull request

5. **Wait for review** - Maintainers will review your PR

## Project Structure

```
ClawSearch/
├── main.py              # Main FastAPI application
├── tests/               # Unit tests
│   ├── __init__.py
│   └── test_main.py
├── .github/
│   └── workflows/
│       └── ci.yml       # CI/CD pipeline
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── Dockerfile          # Container build
├── docker-compose.yml  # Local development
└── README.md           # Project documentation
```

## Reporting Bugs

Please use GitHub Issues to report bugs. Include:

1. **Description** - Clear description of the bug
2. **Steps to reproduce** - Minimal steps to reproduce the issue
3. **Expected behavior** - What you expected to happen
4. **Actual behavior** - What actually happened
5. **Environment** - Python version, OS, etc.

## Feature Requests

Feature requests are welcome! Please:

1. Check if the feature already exists or is planned
2. Provide a clear use case
3. Explain why this feature would be useful

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Follow the Golden Rule

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing! 🎉
