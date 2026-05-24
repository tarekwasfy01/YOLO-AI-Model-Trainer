@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0runtime\python\python.exe"
echo ============================================================
echo Mustatil QGIS runtime check
echo ============================================================
if not exist "%PY%" (
  echo Python runtime missing:
  echo %PY%
  pause
  exit /b 1
)
"%PY%" --version
"%PY%" -c "import numpy, PIL, cv2, onnxruntime, rasterio, geopandas, shapely, pyproj, fiona, ultralytics; print('All imports OK')"
if errorlevel 1 (
  echo.
  echo Import test failed.
  pause
  exit /b 1
)
echo.
echo Runtime OK.
pause