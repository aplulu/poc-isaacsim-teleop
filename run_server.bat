@echo off
rem Start Isaac Sim with the VR teleop scene and the HTTPS/WebRTC server.
rem Extra arguments are passed through, e.g.  run_server.bat --no-stereo --port 9443
rem The web app sends motion as VMC Protocol (OSC) by default; set TELEOP_MOTION_FORMAT=json for the JSON messages.
setlocal
if "%ISAACSIM_PATH%"=="" set ISAACSIM_PATH=C:\isaacsim
cd /d "%~dp0"

if not exist "server\_deps\aiortc" (
  echo [setup] installing Python dependencies into server\_deps ...
  call server\install_deps.bat || exit /b 1
)
if not exist "scene\vr_teleop_scene.usda" (
  echo [setup] building scene\vr_teleop_scene.usda ...
  call "%ISAACSIM_PATH%\python.bat" server\build_scene.py || exit /b 1
)
if not exist "webapp\dist\index.html" (
  echo [setup] building the web app ...
  pushd webapp
  call npm install || exit /b 1
  call npm run build || exit /b 1
  popd
)
call "%ISAACSIM_PATH%\python.bat" server\main.py %*
