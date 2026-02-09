# Check EC2 Status

Check the EC2 server status without deploying.

## Connection Info

- **Host**: 13.124.223.49
- **User**: ec2-user
- **Key**: ~/Documents/AWS/ppt-translator-key.pem
- **Project path**: ~/ppt-translator

## Instructions

1. SSH into EC2 and check status (use single command string, not heredoc, to avoid TTY issues):
   ```bash
   ssh -i ~/Documents/AWS/ppt-translator-key.pem ec2-user@13.124.223.49 'cd ~/ppt-translator && echo "=== Git status ===" && git log --oneline -5 && echo "" && echo "=== Container status ===" && docker compose ps && echo "" && echo "=== Health check ===" && curl -sf http://localhost/health && echo " ✓ Healthy" || echo " ✗ Unhealthy" && echo "" && echo "=== Resource usage ===" && docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"'
   ```

2. Report results to the user concisely.
