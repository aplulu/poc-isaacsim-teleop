@echo off
rem Install the server's extra Python packages next to the project (does not modify C:\isaacsim).
setlocal
if "%ISAACSIM_PATH%"=="" set ISAACSIM_PATH=C:\isaacsim
call "%ISAACSIM_PATH%\python.bat" -m pip install --upgrade --target "%~dp0_deps" -r "%~dp0requirements.txt"
