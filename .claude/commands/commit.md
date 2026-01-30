# Commit Changes

Review staged/unstaged changes and create a well-formed commit.

## Instructions

1. Run `git status` and `git diff` to review all changes
2. Run `git log --oneline -3` to check recent commit message style
3. Stage relevant files (prefer specific files over `git add -A`)
4. Write a concise commit message:
   - First line: imperative mood, under 50 chars (e.g., "Add feature X")
   - Body (if needed): explain "why" not "what"
5. Append co-author line:
   ```
   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   ```
6. Run `git status` after commit to verify

## Safety Rules

- Never amend previous commits unless explicitly requested
- Never skip pre-commit hooks (no `--no-verify`)
- Do not commit `.env`, credentials, or large binaries
- If pre-commit hook fails, fix issues and create a NEW commit
