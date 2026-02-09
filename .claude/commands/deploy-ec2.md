# Deploy to EC2

Deploy the backend to the EC2 production server.

## Connection Info

- **Host**: 13.124.223.49
- **User**: ec2-user
- **Key**: ~/Documents/AWS/ppt-translator-key.pem
- **Project path**: ~/ppt-translator

## Instructions

1. Pre-flight checks (local):
   - Run `git status` to ensure working tree is clean (no uncommitted changes)
   - Run `git log origin/master..HEAD --oneline` to check for unpushed commits
   - If there are unpushed commits, warn the user and ask whether to push first

2. SSH into EC2 and deploy (use single command string, not heredoc, to avoid TTY issues):
   ```bash
   ssh -i ~/Documents/AWS/ppt-translator-key.pem ec2-user@13.124.223.49 'cd ~/ppt-translator && echo "=== Current state ===" && git log --oneline -3 && echo "" && echo "=== Pulling latest changes ===" && git pull origin master && echo "" && echo "=== Rebuilding and restarting containers ===" && docker compose up -d --build && echo "" && echo "=== Waiting for health check ===" && sleep 15 && curl -sf http://localhost/health && echo " ✓ Health check passed" || echo " ✗ Health check failed" && echo "" && echo "=== Container status ===" && docker compose ps'
   ```

3. Report the deployment result to the user:
   - Whether git pull succeeded
   - Whether the build completed
   - Whether the health check passed
   - Current container status

## Troubleshooting

If the health check fails:
- Check container logs: `ssh -i ~/Documents/AWS/ppt-translator-key.pem ec2-user@13.124.223.49 "cd ~/ppt-translator && docker compose logs --tail=50"`
- The container might need more time to start; retry the health check after waiting

## Safety Rules

- Always verify local changes are committed and pushed before deploying
- Never deploy with uncommitted local changes
- Show the user what commits will be deployed (diff between EC2 and latest)
