@echo off
REM =============================================================================
REM Kinship Classification Docker Runner for Windows (Batch Script)
REM =============================================================================
REM
REM Usage:
REM   docker-run.bat build         - Build the Docker image
REM   docker-run.bat train         - Train all models (default settings)
REM   docker-run.bat train 100 32  - Train with 100 epochs, batch size 32
REM   docker-run.bat eval          - Evaluate all models
REM   docker-run.bat shell         - Open interactive shell
REM   docker-run.bat stop          - Stop running containers
REM
REM Requirements:
REM   - Docker Desktop for Windows with WSL2 backend
REM   - NVIDIA GPU with drivers installed
REM   - NVIDIA Container Toolkit
REM =============================================================================

setlocal enabledelayedexpansion

set IMAGE_NAME=kinship-nvidia
set CONTAINER_NAME=kinship-nvidia-runner
set ACTION=%1
set EPOCHS=%2
set BATCH_SIZE=%3
set GPU_ID=%4

if "%EPOCHS%"=="" set EPOCHS=50
if "%BATCH_SIZE%"=="" set BATCH_SIZE=16
if "%GPU_ID%"=="" set GPU_ID=0

if "%ACTION%"=="" (
    echo.
    echo Kinship Classification Docker Runner
    echo =====================================
    echo.
    echo Usage: docker-run.bat [action] [epochs] [batch_size] [gpu_id]
    echo.
    echo Actions:
    echo   build   - Build the Docker image
    echo   train   - Train all models
    echo   eval    - Evaluate all models
    echo   shell   - Open interactive shell
    echo   stop    - Stop running containers
    echo   status  - Show status
    echo.
    echo Examples:
    echo   docker-run.bat build
    echo   docker-run.bat train 100 32 0
    echo   docker-run.bat eval
    echo   docker-run.bat shell
    echo.
    goto :eof
)

if "%ACTION%"=="build" goto :build
if "%ACTION%"=="train" goto :train
if "%ACTION%"=="eval" goto :eval
if "%ACTION%"=="shell" goto :shell
if "%ACTION%"=="stop" goto :stop
if "%ACTION%"=="status" goto :status

echo Unknown action: %ACTION%
goto :eof

:build
echo.
echo ============================================
echo Building Docker Image
echo ============================================
echo.
docker build -f Dockerfile.nvidia -t %IMAGE_NAME% .
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
) else (
    echo.
    echo Build failed!
)
goto :eof

:train
echo.
echo ============================================
echo Training Models
echo ============================================
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo GPU: %GPU_ID%
echo ============================================
echo.

if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints

docker run --gpus all -it --rm ^
    --name %CONTAINER_NAME% ^
    -v "%CD%\data:/app/data" ^
    -v "%CD%\checkpoints:/app/checkpoints" ^
    -v "%CD%\models:/app/models" ^
    -e CUDA_VISIBLE_DEVICES=%GPU_ID% ^
    %IMAGE_NAME% ^
    bash -c "cd /app/models && ./run_all_models_nvidia.sh %EPOCHS% %BATCH_SIZE% %GPU_ID%"
goto :eof

:eval
echo.
echo ============================================
echo Evaluating Models
echo ============================================
echo.

if not exist "results" mkdir results

docker run --gpus all -it --rm ^
    --name %CONTAINER_NAME%-eval ^
    -v "%CD%\data:/app/data:ro" ^
    -v "%CD%\checkpoints:/app/checkpoints:ro" ^
    -v "%CD%\models:/app/models:ro" ^
    -v "%CD%\results:/app/results" ^
    %IMAGE_NAME% ^
    bash -c "cd /app/models && for model_dir in 01_* 02_* 03_* 04_*; do echo Evaluating $model_dir...; if [ -f /app/checkpoints/$model_dir/best.pt ]; then cd $model_dir/Nvidia && python evaluate.py --checkpoint /app/checkpoints/$model_dir/best.pt --output_dir /app/results/$model_dir --full_analysis; cd ../..; else echo No checkpoint for $model_dir; fi; done"
goto :eof

:shell
echo.
echo ============================================
echo Opening Interactive Shell
echo ============================================
echo.
docker run --gpus all -it --rm ^
    --name %CONTAINER_NAME%-shell ^
    -v "%CD%\data:/app/data" ^
    -v "%CD%\checkpoints:/app/checkpoints" ^
    -v "%CD%\models:/app/models" ^
    -v "%CD%\results:/app/results" ^
    %IMAGE_NAME% ^
    bash
goto :eof

:stop
echo.
echo Stopping containers...
docker stop %CONTAINER_NAME% 2>nul
docker stop %CONTAINER_NAME%-eval 2>nul
docker stop %CONTAINER_NAME%-shell 2>nul
echo Done.
goto :eof

:status
echo.
echo ============================================
echo Status
echo ============================================
echo.
echo Docker Containers:
docker ps -a --filter "name=kinship"
echo.
echo GPU Status:
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
goto :eof
