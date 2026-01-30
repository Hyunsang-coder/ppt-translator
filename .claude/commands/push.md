# Push to Remote

Push local commits to the remote repository.

## Instructions

1. Run `git status` to verify:
   - Current branch name
   - Whether branch tracks a remote
   - No uncommitted changes that should be included
2. Run `git log origin/$(git branch --show-current)..HEAD --oneline` to see commits that will be pushed
3. Push with `git push` (or `git push -u origin <branch>` if no upstream)

## Safety Rules

- Never use `--force` or `--force-with-lease` unless explicitly requested
- Never force push to main/master
- Warn user if pushing directly to main/master with multiple commits
