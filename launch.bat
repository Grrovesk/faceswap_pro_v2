@echo off
REM faceswap_pro v2 launcher.
REM Activate your Python venv first (see INSTALL.md for setup),
REM then run this file from inside the v2 directory.
setlocal
cd /d "%~dp0"
python launch.py
pause
endlocal
