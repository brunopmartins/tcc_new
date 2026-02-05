<#
.SYNOPSIS
    Kinship Classification Docker Runner for Windows

.DESCRIPTION
    This script provides easy commands to build and run the kinship classification
    models using Docker on Windows with NVIDIA GPUs.

.PARAMETER Action
    The action to perform: build, train, eval, shell, stop

.PARAMETER Epochs
    Number of training epochs (default: 50)

.PARAMETER BatchSize
    Batch size for training (default: 16)

.PARAMETER GPU
    GPU device ID (default: 0)

.EXAMPLE
    .\docker-run.ps1 build
    .\docker-run.ps1 train -Epochs 100 -BatchSize 32
    .\docker-run.ps1 eval
    .\docker-run.ps1 shell
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("build", "train", "eval", "shell", "stop", "status")]
    [string]$Action,

    [int]$Epochs = 50,
    [int]$BatchSize = 16,
    [int]$GPU = 0
)

$ErrorActionPreference = "Stop"
$ImageName = "kinship-nvidia"
$ContainerName = "kinship-nvidia-runner"

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
}

function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    } catch {
        Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        return $false
    }
}

function Test-NvidiaDocker {
    try {
        docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi | Out-Null
        return $true
    } catch {
        Write-Host "NVIDIA Docker support not available." -ForegroundColor Red
        Write-Host "Make sure you have:" -ForegroundColor Yellow
        Write-Host "  1. Docker Desktop with WSL2 backend" -ForegroundColor Yellow
        Write-Host "  2. NVIDIA drivers installed in Windows" -ForegroundColor Yellow
        Write-Host "  3. NVIDIA Container Toolkit configured" -ForegroundColor Yellow
        return $false
    }
}

switch ($Action) {
    "build" {
        Write-Header "Building Docker Image"

        if (-not (Test-DockerRunning)) { exit 1 }

        Write-Host "Building $ImageName..." -ForegroundColor Green
        docker build -f Dockerfile.nvidia -t $ImageName .

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Build successful!" -ForegroundColor Green
        } else {
            Write-Host "Build failed!" -ForegroundColor Red
            exit 1
        }
    }

    "train" {
        Write-Header "Training Models"

        if (-not (Test-DockerRunning)) { exit 1 }
        if (-not (Test-NvidiaDocker)) { exit 1 }

        # Create directories if they don't exist
        New-Item -ItemType Directory -Force -Path "data" | Out-Null
        New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

        Write-Host "Starting training with:" -ForegroundColor Green
        Write-Host "  Epochs: $Epochs" -ForegroundColor White
        Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
        Write-Host "  GPU: $GPU" -ForegroundColor White
        Write-Host ""

        docker run --gpus all -it --rm `
            --name $ContainerName `
            -v "${PWD}/data:/app/data" `
            -v "${PWD}/checkpoints:/app/checkpoints" `
            -v "${PWD}/models:/app/models" `
            -e CUDA_VISIBLE_DEVICES=$GPU `
            $ImageName `
            bash -c "cd /app/models && ./run_all_models_nvidia.sh $Epochs $BatchSize $GPU"
    }

    "eval" {
        Write-Header "Evaluating Models"

        if (-not (Test-DockerRunning)) { exit 1 }
        if (-not (Test-NvidiaDocker)) { exit 1 }

        New-Item -ItemType Directory -Force -Path "results" | Out-Null

        docker run --gpus all -it --rm `
            --name "${ContainerName}-eval" `
            -v "${PWD}/data:/app/data:ro" `
            -v "${PWD}/checkpoints:/app/checkpoints:ro" `
            -v "${PWD}/models:/app/models:ro" `
            -v "${PWD}/results:/app/results" `
            $ImageName `
            bash -c @"
cd /app/models
for model_dir in 01_* 02_* 03_* 04_*; do
    echo "Evaluating \$model_dir..."
    if [ -f "/app/checkpoints/\$model_dir/best.pt" ]; then
        cd \$model_dir/Nvidia
        python evaluate.py --checkpoint /app/checkpoints/\$model_dir/best.pt --output_dir /app/results/\$model_dir --full_analysis
        cd ../..
    else
        echo "  No checkpoint found for \$model_dir"
    fi
done
"@
    }

    "shell" {
        Write-Header "Opening Interactive Shell"

        if (-not (Test-DockerRunning)) { exit 1 }
        if (-not (Test-NvidiaDocker)) { exit 1 }

        docker run --gpus all -it --rm `
            --name "${ContainerName}-shell" `
            -v "${PWD}/data:/app/data" `
            -v "${PWD}/checkpoints:/app/checkpoints" `
            -v "${PWD}/models:/app/models" `
            -v "${PWD}/results:/app/results" `
            $ImageName `
            bash
    }

    "stop" {
        Write-Header "Stopping Containers"

        docker stop $ContainerName 2>$null
        docker stop "${ContainerName}-eval" 2>$null
        docker stop "${ContainerName}-shell" 2>$null

        Write-Host "Containers stopped." -ForegroundColor Green
    }

    "status" {
        Write-Header "Status"

        if (-not (Test-DockerRunning)) { exit 1 }

        Write-Host "Docker Status:" -ForegroundColor Green
        docker ps -a --filter "name=kinship"

        Write-Host ""
        Write-Host "GPU Status:" -ForegroundColor Green
        docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
    }
}
