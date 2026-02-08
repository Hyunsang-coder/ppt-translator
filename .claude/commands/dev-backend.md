# Start Backend Server

Run the FastAPI backend development server.

## Instructions

1. Check if port 8000 is already in use: `lsof -i :8000`
   - If occupied, inform the user and ask whether to kill the existing process
2. Verify `.env` exists and has at least one API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY)
   - If missing, warn the user
3. Start the server in background:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
4. Wait a few seconds, then verify the server is running via `curl -s http://localhost:8000/health`
5. Report the result to the user

## Notes

- The server runs with `--reload` for auto-restart on code changes
- Default CORS allows `http://localhost:3000` (frontend dev server)
- API docs available at `http://localhost:8000/docs`
