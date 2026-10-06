@echo off
REM Double-click to start the dashboard and the Simulation Engine for a presentation.
cd /d "%~dp0"
py -3.13 -m scripts.start_demo %*
pause
