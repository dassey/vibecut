@echo off
setlocal

set SRC=%~dp0vibecut
set ROOT=%APPDATA%\Blender Foundation\Blender
set FOUND=0

echo Deploying VibeCut addon...
echo   From: %SRC%

for %%V in (5.1 5.2) do (
    if exist "%ROOT%\%%V\scripts\addons\vibecut" (
        set FOUND=1
        echo   To:   %ROOT%\%%V\scripts\addons\vibecut
        xcopy /Y /I "%SRC%\*.py" "%ROOT%\%%V\scripts\addons\vibecut\" >nul
        xcopy /Y /I "%SRC%\*.json" "%ROOT%\%%V\scripts\addons\vibecut\" >nul 2>nul
        xcopy /Y /E /I "%SRC%\Joints" "%ROOT%\%%V\scripts\addons\vibecut\Joints\" >nul
        if exist "%ROOT%\%%V\scripts\addons\vibecut\__pycache__" rd /s /q "%ROOT%\%%V\scripts\addons\vibecut\__pycache__"
    )
)

if "%FOUND%"=="0" (
    echo ERROR: No vibecut addon folder found under %ROOT%. Is Blender installed?
    pause
    exit /b 1
)

echo Done. Reload the addon in Blender (Edit ^> Preferences ^> Add-ons ^> VibeCut ^> disable/enable).
pause
