@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Digital Communication Full-Link Simulation Platform

set "PYEXE="

rem 1) Prefer the verified interpreter on this machine
if exist "D:\office_app\anacond33\envs\python3.10a\python.exe" (
    set "PYEXE=D:\office_app\anacond33\envs\python3.10a\python.exe"
)

rem 2) Fallback: search PATH for a python whose Tk runtime actually works
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE (
            "%%i" -c "import tkinter; w=tkinter.Tk(); w.destroy()" >nul 2>&1
            if not errorlevel 1 set "PYEXE=%%i"
        )
    )
)

if not defined PYEXE (
    echo.
    echo [ERROR] No Python with a working tkinter was found.
    echo Install Python 3.9+ with tcl/tk, then run:
    echo     pip install numpy scipy matplotlib pillow
    echo.
    pause
    exit /b 1
)

echo Using interpreter: %PYEXE%
"%PYEXE%" run_platform.py
if errorlevel 1 (
    echo.
    echo Platform exited with an error. See the message above.
    pause
)
