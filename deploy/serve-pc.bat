@echo off
rem Run the backend on this PC for the Vercel UI (opened on the Pi or anywhere).
rem   deploy\serve-pc.bat https://YOUR-APP.vercel.app [port, default 8003]
rem Or once:  setx CORS_ORIGINS https://YOUR-APP.vercel.app   then just  deploy\serve-pc.bat
rem Binds to 127.0.0.1 only: the public HTTPS address comes from
rem "tailscale funnel --bg 8003" (see DEPLOY.md), not from opening a port.
cd /d "%~dp0.."
if not "%~1"=="" set "CORS_ORIGINS=%~1"
if "%CORS_ORIGINS%"=="" (
    echo Usage: deploy\serve-pc.bat https://YOUR-APP.vercel.app
    exit /b 1
)
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8003"
set "ECOLI_CONFIG=config.yaml"
echo Allowing requests from %CORS_ORIGINS%
".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port %PORT%
