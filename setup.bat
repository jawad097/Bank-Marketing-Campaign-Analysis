@echo off
REM Setup script for Bank Marketing Campaign Analysis project

echo Setting up Bank Marketing Campaign Analysis project...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating necessary directories...
if not exist data mkdir data
if not exist models mkdir models

REM Run data preparation
echo Running data preparation...
python run.py prepare

REM Run model training
echo Running model training...
python run.py train

echo Setup completed successfully!
echo To activate the virtual environment in the future, run: venv\Scripts\activate.bat
echo To run the Streamlit app, run: python run.py app

pause
