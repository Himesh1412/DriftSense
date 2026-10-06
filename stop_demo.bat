@echo off
cd /d "%~dp0"
py -3.13 -m scripts.start_demo --stop
pause
