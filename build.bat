@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=D:\scj\envs\ad\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [WARN] Preferred Python not found: %PYTHON_EXE%
    echo [WARN] Falling back to python from PATH.
    set "PYTHON_EXE=python"
)

echo =============================================
echo   3D Scanner build
echo =============================================
echo Project: %PROJECT_DIR%
echo Python : %PYTHON_EXE%
echo.

cd /d "%PROJECT_DIR%"

echo [1/4] Checking Python...
"%PYTHON_EXE%" --version || goto :fail

echo [2/4] Checking PyInstaller...
"%PYTHON_EXE%" -m PyInstaller --version || goto :fail

echo [3/4] Removing old build outputs...
if exist "%PROJECT_DIR%build" rmdir /s /q "%PROJECT_DIR%build"
if exist "%PROJECT_DIR%dist" rmdir /s /q "%PROJECT_DIR%dist"

echo [4/4] Building executable...
"%PYTHON_EXE%" -m PyInstaller "%PROJECT_DIR%3D_Scanner.spec" --clean --noconfirm || goto :fail

echo.
echo =============================================
echo   Build complete
echo =============================================
echo Output: %PROJECT_DIR%dist\3D_Scanner\3D_Scanner.exe
echo.
exit /b 0

:fail
echo.
echo [ERROR] Build failed.
exit /b 1
