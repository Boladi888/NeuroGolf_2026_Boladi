@echo off
REM VizTerms-style visible-console launcher for the ngs long run.
REM Double-click, or:  start "NGS Longrun" ml\ngs\longrun.bat --tasks hard --name run1 --budget 30 --einsum
REM Close THIS WINDOW to stop the run (it is resumable — re-launch the same command to continue).
title NGS Longrun -- CLOSE THIS WINDOW TO STOP (resumable)
cd /d "<REPO_ROOT>"
if "%~1"=="" (
  echo No args given; using a safe default: --tasks cheap --name run_default --budget 20
  ".venv\Scripts\python.exe" -u -m ml.ngs.longrun --tasks cheap --name run_default --budget 20
) else (
  ".venv\Scripts\python.exe" -u -m ml.ngs.longrun %*
)
echo.
echo   NGS longrun exited. Status/results are under ml\ngs\_runs\. Press any key to close.
pause >nul
