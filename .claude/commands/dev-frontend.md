# Start Frontend Server

Run the Next.js frontend development server.

## Instructions

1. Check if port 3000 is already in use: `lsof -i :3000`
   - If occupied, inform the user and ask whether to kill the existing process
2. Check if `frontend/node_modules` exists
   - If not, run `cd frontend && npm install` first
3. Start the server in background:
   ```bash
   cd frontend && npm run dev
   ```
4. Wait a few seconds, then verify the server is running via `curl -s http://localhost:3000`
5. Report the result to the user

## Notes

- Requires backend running on port 8000 for full functionality
- Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` for local development
- Frontend works with graceful fallback (limited features) when backend is unavailable
