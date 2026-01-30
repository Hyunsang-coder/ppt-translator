# Update CLAUDE.md Documentation

Analyze the current codebase and update CLAUDE.md to reflect the latest project structure, patterns, and conventions.

## Instructions

1. **Gather Current State**
   - Read the existing CLAUDE.md
   - Scan the project structure: `src/`, `tests/`, root files
   - Identify new modules, removed files, or renamed components

2. **Analyze Key Areas**
   - Entry points: `app.py`, `api.py`, CLI scripts
   - Architecture: `src/core/`, `src/chains/`, `src/services/`, `src/utils/`, `src/ui/`
   - Dependencies: `requirements.txt`, `pyproject.toml`
   - Configuration: `.env.example`, environment variables
   - Tests: `tests/` structure and patterns

3. **Update Sections**
   - **Project Overview**: Ensure it reflects current functionality
   - **Development Commands**: Verify all commands still work
   - **Environment Setup**: Check for new env variables
   - **Architecture**: Add new modules, remove obsolete ones
   - **Key Patterns**: Document new patterns, update changed flows
   - **Libraries**: Sync with requirements.txt

4. **Quality Checks**
   - Remove references to deleted files/functions
   - Add documentation for new public APIs
   - Ensure code examples are accurate
   - Keep descriptions concise but complete

5. **Output**
   - Show a summary of changes made
   - Do NOT commit automatically - let user review first

## Guidelines

- Preserve existing structure and formatting style
- Focus on information useful for AI assistants (Claude Code)
- Include file paths with line references for key functions
- Document non-obvious patterns and conventions
- Keep Korean/English consistent with existing content
