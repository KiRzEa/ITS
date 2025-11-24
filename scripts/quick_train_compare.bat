@echo off
REM Quick Start Script for Training and Comparing All Models (Windows)
REM This script automates the entire process from data preparation to visualization

setlocal enabledelayedexpansion

echo ==================================
echo OBJECT DETECTION MODEL COMPARISON
echo Quick Start Script (Windows)
echo ==================================
echo.

REM Configuration
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=20
set DATA_ROOT=data/raw/yolov8
set COCO_DIR=data/processed/coco_format
set EXPERIMENTS_DIR=experiments/model_comparison

echo Configuration:
echo   Epochs: %EPOCHS%
echo   Data root: %DATA_ROOT%
echo   COCO annotations: %COCO_DIR%
echo   Experiments dir: %EXPERIMENTS_DIR%
echo.

REM Step 1: Check if data exists
echo Step 1/3: Checking data...
if not exist "%DATA_ROOT%\train\images" (
    echo Error: Training data not found at %DATA_ROOT%\train\images
    echo Please ensure your data is in the correct location.
    exit /b 1
)
echo OK Data found
echo.

REM Step 2: Prepare COCO format annotations
echo Step 2/3: Preparing COCO format annotations...
if not exist "%COCO_DIR%\train_coco.json" (
    echo Converting YOLO format to COCO format...
    python scripts/prepare_coco_format.py --data-root "%DATA_ROOT%" --output-dir "%COCO_DIR%"
) else (
    echo OK COCO format annotations already exist
)
echo.

REM Step 3: Train and compare all models
echo Step 3/3: Training and comparing all models...
echo This will take several hours depending on your GPU.
echo Press Ctrl+C to cancel, or wait 5 seconds to continue...
timeout /t 5 /nobreak

python scripts/train_and_compare_all_models.py --epochs %EPOCHS% --data-root "%DATA_ROOT%" --coco-annotations "%COCO_DIR%" --experiments-dir "%EXPERIMENTS_DIR%"

REM Done
echo.
echo ==================================
echo COMPARISON COMPLETE!
echo ==================================
echo.
echo Results saved to: %EXPERIMENTS_DIR%
echo.
echo View your results:
echo   - Comparison plots: %EXPERIMENTS_DIR%\model_comparison.png
echo   - Training curves: %EXPERIMENTS_DIR%\training_curves.png
echo   - Detailed report: %EXPERIMENTS_DIR%\comparison_report.txt
echo   - JSON results: %EXPERIMENTS_DIR%\all_results.json
echo.
echo To view the comparison plot:
echo   start %EXPERIMENTS_DIR%\model_comparison.png
echo.

pause
