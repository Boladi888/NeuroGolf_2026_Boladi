@echo off
REM Open a second visible window that watches a running longrun's live dashboard.
REM Usage:  start "NGS Watch" ml\ngs\watch.bat run1      (run1 = the --name of the longrun)
title NGS Longrun WATCH
cd /d "<REPO_ROOT>"
set NAME=%~1
if "%NAME%"=="" set NAME=run_default
".venv\Scripts\python.exe" -u -m ml.ngs.longrun --watch --name %NAME%
echo.
pause >nul
