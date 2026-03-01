@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating directories...
if not exist "data" mkdir data
if not exist "vector_store" mkdir vector_store
if not exist "logs" mkdir logs

echo.
echo Copying environment file...
if not exist ".env" copy .env.example .env

echo.
echo ========================================
echo Setup completed!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Place your CSV data in ./data/comments.csv
echo    (or run: python scripts/generate_sample_data.py)
echo 3. Build knowledge base: python scripts/build_knowledge_base.py
echo 4. Start services: start.bat
echo ========================================
