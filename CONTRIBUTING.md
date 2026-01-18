# Contributing to Vexa Polish LLM

Thank you for your interest in contributing to Vexa Polish LLM! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/ENERQU4489/vexa-polish-llm.git
   cd vexa-polish-llm
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ENERQU4489/vexa-polish-llm.git
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool

### Setup Steps

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

This installs:
- `pytest` - Testing framework
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker
- `isort` - Import sorter

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Python version** and OS
- **Error messages** or logs
- **Screenshots** if applicable

**Bug Report Template**:
```markdown
**Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Vexa version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title** and detailed description
- **Use case** - why is this enhancement useful?
- **Proposed solution** - how should it work?
- **Alternatives considered**
- **Additional context** - mockups, examples, etc.

### Contributing Code

1. **Find or create an issue** to work on
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following coding standards
5. **Write tests** for new functionality
6. **Run tests** to ensure nothing breaks
7. **Commit your changes** with clear messages
8. **Push to your fork** and create a pull request

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Single quotes for strings (except docstrings)
- **Docstrings**: Google style

### Code Formatting

Use `black` for automatic formatting:

```bash
black src/ tests/
```

### Linting

Run `flake8` before committing:

```bash
flake8 src/ tests/
```

### Type Hints

Use type hints for function signatures:

```python
def process_text(text: str, max_length: int = 100) -> List[str]:
    """Process text and return list of tokens."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(data: List[str], epochs: int = 100) -> None:
    """
    Train the ACO model on provided data.
    
    Args:
        data: List of training texts
        epochs: Number of training epochs
        
    Returns:
        None
        
    Raises:
        ValueError: If data is empty
    """
    pass
```

### Import Organization

Use `isort` to organize imports:

```bash
isort src/ tests/
```

Import order:
1. Standard library
2. Third-party packages
3. Local modules

```python
import os
import sys

import numpy as np
import yaml

from src.core import AntGraph
from src.utils import VexaTokenizer
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tokenizer.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

Example:

```python
def test_tokenizer_encodes_polish_characters():
    """Test that tokenizer correctly encodes Polish diacritics."""
    tokenizer = VexaTokenizer()
    tokenizer.build_vocab(['ąćęłńóśźż'], min_freq=1)
    
    encoded = tokenizer.encode('ąć')
    assert len(encoded) > 0
    assert all(isinstance(x, int) for x in encoded)
```

### Test Coverage

Aim for:
- **80%+ coverage** for new code
- **100% coverage** for critical paths
- Tests for edge cases and error conditions

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run all tests** and ensure they pass
4. **Format code** with black
5. **Check linting** with flake8
6. **Update CHANGELOG.md** if applicable

### PR Title Format

Use conventional commit format:

- `feat: Add new feature`
- `fix: Fix bug in tokenizer`
- `docs: Update README`
- `test: Add tests for ACO algorithm`
- `refactor: Improve code structure`
- `perf: Optimize training loop`

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one approval** from maintainer
3. **All conversations resolved**
4. **Up to date** with main branch

### After Approval

Maintainers will:
1. Merge your PR
2. Update version if needed
3. Deploy changes (if applicable)

## Development Workflow

### Typical Workflow

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Format and lint
black src/
flake8 src/

# 5. Run tests
pytest

# 6. Commit changes
git add .
git commit -m "feat: Add my feature"

# 7. Push to fork
git push origin feature/my-feature

# 8. Create PR on GitHub
```

### Keeping Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## Areas for Contribution

### High Priority

- 🔥 Performance optimization
- 🔥 Additional tests
- 🔥 Documentation improvements
- 🔥 Web interface (Flask/FastAPI)

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation fixes
- Code comments
- Simple bug fixes
- Test additions

### Advanced Contributions

- Algorithm improvements
- New features
- Architecture changes
- Performance optimization

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainers directly

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to Vexa Polish LLM! 🐜🇵🇱
