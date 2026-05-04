@echo off
echo Starting CCTV Supervisor...

:: Change to the directory where this batch script is located
cd /d "%~dp0"

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Run the script
python main.py --duration_loop 30

:: --display true

:: Keep the window open if the script crashes or closes
pause
