@echo off
REM Daily paper bot launcher for Windows Task Scheduler.
REM Keep this file ASCII-only: .bat is read in the OEM code page, and %~dp0
REM expands the real path at run time, so a CJK project folder stays safe.

REM Task Scheduler may start with its own working directory; the config uses
REM paths relative to the project root.
cd /d "%~dp0.."

REM Without this the console falls back to the ANSI code page (cp950 on zh-TW)
REM and a paper title outside it aborts the run with UnicodeEncodeError.
set PYTHONUTF8=1

if not exist logs mkdir logs
.venv\Scripts\python.exe main.py >> logs\daily.log 2>&1
