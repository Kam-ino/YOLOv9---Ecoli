@echo off
REM run.bat — start backend + frontend with one command.
REM
REM   - backend opens in a new window titled "ecoli-backend"
REM     (http://localhost:8003)
REM   - frontend runs in this window (http://localhost:3003)
REM
REM Ctrl+C in this window stops Vite; close the backend window to stop
REM uvicorn. Edit ECOLI_CONFIG below to switch between the trained
REM model (config.yaml) and the smoke-test COCO model (config.smoke.yaml).

set ECOLI_CONFIG=config.yaml

start "ecoli-backend" cmd /k "set ECOLI_CONFIG=%ECOLI_CONFIG% && .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8003"

cd frontend
call npm run dev
