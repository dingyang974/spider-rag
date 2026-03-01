@echo off
echo ========================================
echo Starting Backend API Server...
echo ========================================
start cmd /k "cd /d %~dp0 && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak > nul

echo ========================================
echo Starting Frontend Streamlit App...
echo ========================================
start cmd /k "cd /d %~dp0 && streamlit run app.py"

echo ========================================
echo Services started!
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:8501
echo ========================================
