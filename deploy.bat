@echo off
setlocal EnableDelayedExpansion

set "SRC=%~dp0vibecut"
set "ROOT=%APPDATA%\Blender Foundation\Blender"
set "PF=%ProgramFiles%\Blender Foundation"
set FOUND=0

echo Deploying VibeCut addon...
echo   From: %SRC%

rem Candidate versions: every user config folder, plus every installed
rem "Blender X.Y" (a fresh install has no config folder until first launch).
set "VERSIONS="
if exist "%ROOT%" for /d %%D in ("%ROOT%\*") do set "VERSIONS=!VERSIONS! %%~nxD"
if exist "%PF%" for /d %%D in ("%PF%\Blender *") do (
    set "N=%%~nxD"
    set "VERSIONS=!VERSIONS! !N:Blender =!"
)

for %%V in (!VERSIONS!) do call :deploy %%V

if "%FOUND%"=="0" (
    echo ERROR: No Blender 5.1 or newer found under %ROOT% or %PF%.
    pause
    exit /b 1
)

echo Done. Reload the addon in Blender (Edit ^> Preferences ^> Add-ons ^> VibeCut ^> disable/enable).
pause
exit /b 0


:deploy
set "V=%~1"
if defined SEEN_%V% goto :eof
set "SEEN_%V%=1"
set "MAJ=" & set "MIN="
for /f "tokens=1,2 delims=." %%a in ("%V%") do (
    set "MAJ=%%a"
    set "MIN=%%b"
)
echo(%MAJ%| findstr /r "^[0-9][0-9]*$" >nul || goto :eof
echo(%MIN%| findstr /r "^[0-9][0-9]*$" >nul || goto :eof
set OK=0
if %MAJ% GTR 5 set OK=1
if %MAJ% EQU 5 if %MIN% GEQ 1 set OK=1
if "%OK%"=="0" goto :eof

set "RES=%ROOT%\%V%"
set "DST=%RES%\scripts\addons\vibecut"
set NEW=0
if not exist "%DST%" (
    set "NEW=1"
    mkdir "%DST%"
)
set FOUND=1
echo   To:   %DST%
xcopy /Y /I "%SRC%\*.py" "%DST%\" >nul
xcopy /Y /I "%SRC%\*.json" "%DST%\" >nul 2>nul
xcopy /Y /E /I "%SRC%\Joints" "%DST%\Joints\" >nul
rem Stale .pyc files shadow freshly copied sources on reload.
if exist "%DST%\__pycache__" rd /s /q "%DST%\__pycache__"

if "%NEW%"=="1" call :enable
goto :eof


:enable
rem First install for this version: enable the addon and save preferences.
set "EXE=%PF%\Blender %V%\blender.exe"
if not exist "%EXE%" (
    echo         New install: open Blender %V% and enable VibeCut in Preferences ^> Add-ons.
    goto :eof
)
set "BLENDER_USER_RESOURCES=%RES%"
"%EXE%" -b --python-exit-code 1 --python-expr "import bpy; bpy.ops.preferences.addon_enable(module='vibecut'); bpy.ops.wm.save_userpref()" >nul 2>&1
set "RC=!errorlevel!"
set "BLENDER_USER_RESOURCES="
if not "%RC%"=="0" (
    echo         Could not enable automatically: enable VibeCut in Blender %V% Preferences ^> Add-ons.
) else (
    echo         Enabled VibeCut in Blender %V%.
)
goto :eof
