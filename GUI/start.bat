@echo off
rem NeuroGolf Lab - opens the server in its own visible console window.
rem Closing that window kills the server (in-process uvicorn + a
rem KILL_ON_JOB_CLOSE JobObject reap any strays). No orphans.
cd /d "%~dp0"

if not exist client\dist (
  echo client\dist missing - building the web client ^(one-time^)...
  pushd client
  call npm install && call npm run build
  popd
)

start "NeuroGolf Lab" cmd /k python launch.py
echo NeuroGolf Lab window opened. Close that window to stop the server.
