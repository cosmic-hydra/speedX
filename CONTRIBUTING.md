# Contributing to speedX

Thank you for your interest in contributing to speedX! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/cosmic-hydra/speedX.git
cd speedX
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=speedx --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_classifier.py -v
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings in NumPy style
- Keep functions focused and modular
- Add tests for new functionality

## Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your feature with tests

3. Ensure all tests pass:
```bash
pytest
```

4. Update documentation if needed

5. Submit a pull request

## Testing Guidelines

- Write tests for all new functionality
- Aim for high code coverage (>80%)
- Use descriptive test names
- Include edge cases
- Mock external dependencies (e.g., HTTP requests)

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update CHANGELOG.md

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Submit PR with clear description
5. Address review comments

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about usage
- Suggestions for improvements

Thank you for contributing!
