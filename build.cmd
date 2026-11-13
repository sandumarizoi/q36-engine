@echo off
setlocal
rem Builds q36 for Windows x64. Output: build\ (objects) and bin\ (executables + the oneAPI runtime DLLs they need).
rem Requirements: Visual Studio 2022 Build Tools (C++ workload), Intel oneAPI DPC++/C++ Compiler 2025.x or newer.
rem Optional environment variables:
rem   ONEAPI_ROOT   oneAPI install folder        (default: C:\Program Files (x86)\Intel\oneAPI)
rem   SETVARS_ARGS  extra arguments for setvars  (e.g. --config=my.cfg to pick component versions)
rem   VSVARS        path to vcvars64.bat          (default: VS 2022 Build Tools)
if "%ONEAPI_ROOT%"=="" set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if "%VSVARS%"=="" set "VSVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSVARS%" goto novs
if not exist "%ONEAPI_ROOT%\setvars.bat" goto nooneapi
call "%VSVARS%" >nul
call "%ONEAPI_ROOT%\setvars.bat" %SETVARS_ARGS% --force intel64 >nul
set "ROOT=%~dp0"
if not exist "%ROOT%build" mkdir "%ROOT%build"
if not exist "%ROOT%bin" mkdir "%ROOT%bin"
cd /d "%ROOT%build"

rem GPU programs: SYCL + ESIMD, kernels compiled ahead of time for the Arc B580 (Xe2, bmg-g21)
set GPUFLAGS=-D_CRT_SECURE_NO_WARNINGS -fsycl -O2 /arch:AVX2 /EHsc -fsycl-targets=spir64_gen -Xs "-device bmg-g21"
echo [1/4] q36-server
icx %GPUFLAGS% ..\tools\server.cpp -o q36-server.exe ws2_32.lib || goto fail
echo [2/4] q36 (CLI)
icx %GPUFLAGS% ..\tools\q36.cpp -o q36.exe || goto fail
rem CPU-only tools
echo [3/4] pack
icx /nologo /O2 /std:c++17 /arch:AVX2 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS /Fe:pack.exe ..\tools\pack.cpp || goto fail
echo [4/4] mktok
icx /nologo /O2 /std:c++17 /arch:AVX2 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS /Fe:mktok.exe ..\tools\mktok.cpp || goto fail

copy /y q36-server.exe "%ROOT%bin" >nul & copy /y q36.exe "%ROOT%bin" >nul & copy /y pack.exe "%ROOT%bin" >nul & copy /y mktok.exe "%ROOT%bin" >nul
rem oneAPI runtime next to the executables, so bin\ runs without setvars
for %%p in (sycl8.dll sycl9.dll sycl10.dll ur_loader.dll ur_loader_shared.dll ur_win_proxy_loader.dll ur_adapter_level_zero.dll ur_adapter_level_zero_v2.dll svml_dispmd.dll libmmd.dll libiomp5md.dll libirngmd.dll libimalloc.dll umf.dll tbb12.dll tbbmalloc.dll tcm.dll libhwloc-15.dll) do (
  for /f "delims=" %%f in ('where %%p 2^>nul') do ( if not exist "%ROOT%bin\%%~nxf" copy /y "%%f" "%ROOT%bin" >nul )
)
echo build ok: %ROOT%bin
exit /b 0
:fail
echo build failed
exit /b 1
:novs
echo vcvars64.bat not found: "%VSVARS%" - set VSVARS
exit /b 1
:nooneapi
echo oneAPI not found: "%ONEAPI_ROOT%" - set ONEAPI_ROOT
exit /b 1
