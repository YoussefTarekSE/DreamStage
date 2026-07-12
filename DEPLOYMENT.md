# DreamStage Deployment

## Backend

1. Create a Python 3.12 environment.
2. Install dependencies:
   ```powershell
   cd backend
   pip install -r requirements.txt
   ```
3. Configure environment variables from `.env.example`.
4. Run Supabase migrations:
   ```powershell
   cd C:\path\to\dreamstage
   $env:SUPABASE_MANAGEMENT_TOKEN="..."
   $env:SUPABASE_PROJECT_REF="..."
   .\scripts\migrate.ps1
   ```
5. Start the API:
   ```powershell
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

Render can deploy the backend from `render.yaml`. Store all secrets in the Render dashboard; do not commit `.env` files.

## Frontend

1. Install Node dependencies:
   ```powershell
   cd frontend
   npm ci
   ```
2. Configure frontend variables from `frontend/.env.example`.
3. Build or run:
   ```powershell
   npm run lint
   npx tsc --noEmit
   npm run build
   ```

## Monitoring

- `GET /health` reports app, database, storage, and AI subsystem status.
- `GET /metrics` returns in-process generation, transcription, mixing, and cache metrics.
- `GET /admin/beat-metrics` returns persisted beat telemetry and requires `ADMIN_KEY`.
