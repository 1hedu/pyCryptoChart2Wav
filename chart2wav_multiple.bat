@echo off
cd /d "%~dp0"
set "PY=..\Inkscape\python.exe"
set "SCRIPT=chart2wav\chart_png_to_wave.py"

for %%F in (chart2wav\2025-11-13_WPLS-pWETH.png) do (
    echo Processing %%F ...
    "%PY%" "%SCRIPT%" "%%F" "%%~nF.wav" --mode multi --cycles 4 --points 256 --seconds 1.0
)

echo Done!
pause
