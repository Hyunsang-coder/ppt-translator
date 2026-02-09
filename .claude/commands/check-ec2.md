# Check EC2 Status

Check the EC2 server status without deploying.

## Connection Info

- **Host**: 13.124.223.49
- **User**: ec2-user
- **Key**: ~/Documents/AWS/ppt-translator-key.pem
- **Project path**: ~/ppt-translator

## Instructions

1. SSH into EC2 and check status:
   ```bash
   ssh -i ~/Documents/AWS/ppt-translator-key.pem ec2-user@13.124.223.49 << 'CHECK'
   echo "=== Git status ==="
   cd ~/ppt-translator
   git log --oneline -5
   echo ""
   echo "=== Container status ==="
   docker compose ps
   echo ""
   echo "=== Health check ==="
   if curl -sf http://localhost:8000/health; then echo " ✓ Healthy"; else echo " ✗ Unhealthy"; fi
   echo ""
   echo "=== Resource usage ==="
   docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true
   CHECK
   ```

2. Report results to the user concisely.
