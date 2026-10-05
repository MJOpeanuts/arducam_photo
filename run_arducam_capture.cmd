@echo off
rem Launches the graphical app with the installed environment; no PowerShell activation needed.
rem Expects the venv in .venv next to this file (see README); falls back to "py -3" otherwise.
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\pythonw.exe" (
  start "" "%HERE%.venv\Scripts\pythonw.exe" -m arducam_photo.app %*
) else (
  py -3 -m arducam_photo.app %*
)
