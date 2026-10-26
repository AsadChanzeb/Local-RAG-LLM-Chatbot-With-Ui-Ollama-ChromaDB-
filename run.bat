@echo off
echo ==============================================
echo   Starting Local RAG Chatbot (Ollama + Chroma)
echo ==============================================
if exist "%~dp0venv311\Scripts\python.exe" (
    "%~dp0venv311\Scripts\python.exe" app.py
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
