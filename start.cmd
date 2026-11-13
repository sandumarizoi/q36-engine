@echo off
rem Starts the q36 server: OpenAI-compatible API on http://127.0.0.1:9936/v1
rem MODEL = the converted .q36 file (the .tok file with the same name must be next to it). Extra arguments are passed through,
rem e.g.  start.cmd --ctx 65536 --port 8080
if "%MODEL%"=="" set "MODEL=%~dp0models\Qwen3.6-35B-A3B-UD-Q4_K_M.q36"
if not exist "%MODEL%" goto nomodel
cd /d "%~dp0bin"
q36-server.exe "%MODEL%" --ctx 32768 --vram-gb 10.5 --mtp 2 --port 9936 %*
pause
exit /b 0
:nomodel
echo Model not found: "%MODEL%"
echo Set MODEL or put the files into the models folder.
pause
exit /b 1
