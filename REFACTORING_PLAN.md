# Refactoring Plan for GitHub Release

## Objectives
1. Remove all Polish comments and replace with English
2. Clean up file structure
3. Update documentation (README, TODO, requirements)
4. Add proper GitHub files (.github, CODE_OF_CONDUCT, etc.)
5. Remove test files created during debugging

## Files to Refactor

### Core Files (Polish → English)
- [ ] src/core/graph.py
- [ ] src/core/engine.py
- [ ] src/core/agent.py
- [ ] src/integration/llm_interface.py
- [ ] src/utils/tokenizer.py
- [ ] src/utils/cleaner.py
- [ ] src/utils/sharder.py
- [ ] src/utils/wiki_downloader.py
- [ ] main.py

### Documentation Files
- [ ] README.md - Complete rewrite
- [ ] TODO.md - Update and translate
- [ ] requirements.txt - Verify and clean
- [ ] requirements-dev.txt - Verify and clean
- [ ] CONTRIBUTING.md - Update
- [ ] QUICKSTART.md - Translate
- [ ] INSTALL.md - Translate

### Files to Remove
- [ ] test_chat_fixes.py (debugging file)
- [ ] test_integration.py (debugging file)
- [ ] quick_chat_test.py (debugging file)
- [ ] TODO_fix.md (temporary file)
- [ ] REFACTORING_PLAN.md (this file, after completion)

### Files to Add
- [ ] .github/ISSUE_TEMPLATE/bug_report.md
- [ ] .github/ISSUE_TEMPLATE/feature_request.md
- [ ] .github/workflows/tests.yml (CI/CD)
- [ ] CODE_OF_CONDUCT.md
- [ ] CHANGELOG.md

## Order of Operations
1. Remove temporary test files
2. Refactor core Python files (comments, docstrings)
3. Update documentation files
4. Add GitHub-specific files
5. Final review and cleanup
