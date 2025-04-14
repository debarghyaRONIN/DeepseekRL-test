@echo off
echo Installing dependencies for Pokemon Red with Ollama deepseek-r1
echo.

REM Activate the conda environment
call conda activate main

REM Check if activation was successful
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate conda environment 'main'
    echo.
    echo Would you like to create the 'main' environment? (y/n)
    set /p create_env=
    if /i "%create_env%"=="y" (
        echo Creating conda 'main' environment...
        call conda create -n main python=3.9 -y
        call conda activate main
    ) else (
        echo Exiting without installing dependencies.
        goto :EOF
    )
)

echo Installing required packages in the conda 'main' environment...

REM Install packages one by one to better handle errors
call conda install -n main -y numpy
call conda install -n main -y scikit-image
call conda install -n main -y -c conda-forge gym
call conda install -n main -y -c conda-forge stable-baselines3
call conda install -n main -y pygame
call conda install -n main -y pillow
call conda install -n main -y tensorboard
call conda install -n main -y requests

echo.
echo Dependency installation complete!
echo You can now run the Ollama Pokemon agent using run_with_conda.bat
pause 