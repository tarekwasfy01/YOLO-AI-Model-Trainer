@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "RUNTIME=%~dp0runtime"
set "PYDIR=%RUNTIME%\python"
set "PYEXE=%PYDIR%\python.exe"

echo ============================================================
echo Mustatil QGIS runtime installer
echo ============================================================

if not exist "%RUNTIME%" mkdir "%RUNTIME%"

REM ============================================================
REM FIRST: Try to use already installed system Python
REM ============================================================

set "SYSTEMPY="

where py.exe >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "SYSTEMPY=%%i"
    )
)

if "%SYSTEMPY%"=="" (
    where python.exe >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            set "SYSTEMPY=%%i"
        )
    )
)

if not "%SYSTEMPY%"=="" (
    echo.
    echo Existing Python installation detected:
    echo %SYSTEMPY%
    echo.
    echo Installing dependencies into existing Python...
    "%SYSTEMPY%" -m pip install --upgrade pip
    "%SYSTEMPY%" -m pip install numpy pillow opencv-python onnxruntime rasterio geopandas shapely pyproj fiona ultralytics torch torchvision pyyaml --index-url https://download.pytorch.org/whl/cpu
    "%SYSTEMPY%" -m pip install ultralytics

    if errorlevel 1 (
        echo.
        echo Dependency installation into existing Python failed.
        echo Falling back to embedded runtime...
    ) else (
        echo.
        echo ============================================================
        echo Existing Python configured successfully.
        echo ============================================================

        REM write detected python path to runtime_python.txt
        echo %SYSTEMPY% > "%~dp0runtime_python.txt"

        pause
        exit /b 0
    )
)

echo.
echo No usable existing Python found. Using embedded runtime.

set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
set "GETPIPURL=https://bootstrap.pypa.io/get-pip.py"
set "PYZIP=%RUNTIME%\python312_embed.zip"
set "GETPIP=%RUNTIME%\get-pip.py"

if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
  set "PWSH=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
) else (
  set "PWSH="
)

if not exist "%PYEXE%" (

  echo Downloading embedded Python 3.12.10...

  if defined PWSH (
    "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri '%PYURL%' -OutFile '%PYZIP%' -UseBasicParsing } catch { exit 1 }"
  )

  if not exist "%PYZIP%" (
    where curl.exe >nul 2>nul
    if not errorlevel 1 (
      curl.exe -L "%PYURL%" -o "%PYZIP%"
    )
  )

  if not exist "%PYZIP%" (
    bitsadmin /transfer MustatilPythonDownload /download /priority normal "%PYURL%" "%PYZIP%"
  )

  if not exist "%PYZIP%" (
    echo ERROR: Could not download Python automatically.
    pause
    exit /b 1
  )

  echo Extracting Python...

  if defined PWSH (
    "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%PYZIP%' '%PYDIR%'"
  ) else (
    tar -xf "%PYZIP%" -C "%PYDIR%"
  )

  if not exist "%PYEXE%" (
    echo ERROR: Python extraction failed.
    pause
    exit /b 1
  )
)

echo Enabling site-packages...

if exist "%PYDIR%\python312._pth" (
  if defined PWSH (
    "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content '%PYDIR%\python312._pth') -replace '#import site','import site' | Set-Content '%PYDIR%\python312._pth'"
  ) else (
    echo import site>>"%PYDIR%\python312._pth"
  )
)

if not exist "%PYDIR%\Scripts\pip.exe" (

  echo Downloading get-pip.py...

  if defined PWSH (
    "%PWSH%" -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri '%GETPIPURL%' -OutFile '%GETPIP%' -UseBasicParsing } catch { exit 1 }"
  )

  if not exist "%GETPIP%" (
    where curl.exe >nul 2>nul
    if not errorlevel 1 (
      curl.exe -L "%GETPIPURL%" -o "%GETPIP%"
    )
  )

  if not exist "%GETPIP%" (
    bitsadmin /transfer MustatilPipDownload /download /priority normal "%GETPIPURL%" "%GETPIP%"
  )

  if not exist "%GETPIP%" (
    echo ERROR: Could not download get-pip.py
    pause
    exit /b 1
  )

  echo Installing pip...
  "%PYEXE%" "%GETPIP%"
)

echo Installing dependencies into embedded runtime...

"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install numpy pillow opencv-python onnxruntime rasterio geopandas shapely pyproj fiona ultralytics torch torchvision pyyaml --index-url https://download.pytorch.org/whl/cpu
"%PYEXE%" -m pip install ultralytics

if errorlevel 1 (
    echo.
    echo Runtime dependency installation failed.
    pause
    exit /b 1
)

echo %PYEXE% > "%~dp0runtime_python.txt"

echo.
echo ============================================================


echo.
echo Preparing SAM2 runtime files...
"%PYEXE%" "%~dp0scripts\mustatil_sam2_runtime_setup.py" --dest "%~dp0runtime\sam2" --model tiny
echo SAM2 runtime setup command finished.

echo Runtime installed successfully.
echo ============================================================

pause
exit /b 0

echo.
echo SAM2 optional:
echo Real SAM2 requires installing the official SAM2 package into this runtime
echo and selecting a checkpoint/config in the Annotator tab.
echo.
