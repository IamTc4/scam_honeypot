@echo off
TITLE Competition Day - ngrok Static Domain Startup
CLS

ECHO ================================================
ECHO   COMPETITION DAY - ngrok Static Domain Startup
ECHO   Stable URL that NEVER changes!
ECHO ================================================
ECHO.

:: Check ngrok
where ngrok >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO [!] ngrok not found.
    ECHO Install: winget install ngrok.ngrok
    ECHO Or download from: https://ngrok.com/download
    ECHO.
    ECHO After installing, run:
    ECHO   ngrok config add-authtoken YOUR_TOKEN
    PAUSE
    EXIT /B
)

ECHO [OK] ngrok found.
ECHO.

:: Check for static domain
IF "%NGROK_STATIC_DOMAIN%"=="" (
    SET NGROK_STATIC_DOMAIN=pixilated-spraylike-zona.ngrok-free.dev
)

ECHO [OK] Static domain: %NGROK_STATIC_DOMAIN%
ECHO.

:: Cleanup
ECHO Cleaning up old processes...
taskkill /F /IM python.exe >nul 2>nul
taskkill /F /IM ngrok.exe >nul 2>nul
timeout /t 2 /nobreak >nul

:: Start Server
ECHO Starting local server on port 8001...
start "Scam Honeypot Server" python -m uvicorn src.main:app --host 0.0.0.0 --port 8001
timeout /t 5 /nobreak >nul

:: Start ngrok with static domain
ECHO Starting ngrok with static domain...
start "ngrok Tunnel" ngrok http 8001 --domain=%NGROK_STATIC_DOMAIN%

ECHO ================================================
ECHO   SYSTEM RUNNING!
ECHO ================================================
ECHO.
ECHO Your PERMANENT URL: https://%NGROK_STATIC_DOMAIN%
ECHO API Endpoint: https://%NGROK_STATIC_DOMAIN%/api/scam-honeypot
ECHO.
ECHO Set LOCAL_TUNNEL_URL in HF Secrets to:
ECHO   https://%NGROK_STATIC_DOMAIN%
ECHO.
ECHO (Only need to set this ONCE - URL never changes!)
ECHO.
ECHO Press any key to stop everything...
PAUSE

taskkill /F /IM python.exe >nul 2>nul
taskkill /F /IM ngrok.exe >nul 2>nul
ECHO Stopped.
