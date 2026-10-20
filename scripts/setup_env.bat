@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
call conda env list
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n testmodel python -c "import sys; print(sys.executable); assert sys.prefix.lower().endswith('testmodel')"
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n testmodel python -m scripts.install_dependencies
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n testmodel python -m scripts.check_env --config configs/sddnet_v1_100e.yaml
if errorlevel 1 exit /b 1
echo Environment setup and CUDA verification passed.
exit /b 0
