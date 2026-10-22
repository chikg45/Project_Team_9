@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
    set "PYTHON=python"
) else (
    set "PYTHON=py -3"
)
%PYTHON% --version
if errorlevel 1 (
    echo Can cai Python truoc khi chay.
    pause
    exit /b 1
)
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Cai thu vien chua thanh cong. Kiem tra ket noi mang.
    pause
    exit /b 1
)
%PYTHON% -m streamlit run app\AI.py
pause
