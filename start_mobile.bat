@echo off
echo ========================================
echo Thai-English Translator - Mobile
echo Solo + Room mode (QR code)
echo ========================================
echo.

cd /d "%~dp0mobile"

echo Installing/updating npm dependencies...
npm install

echo.
echo ========================================
echo Starting Expo with tunnel...
echo ========================================
echo.
echo Before scanning the QR code, set BACKEND_URL in mobile/App.js
echo to your ngrok (or cloudflared) HTTPS URL.
echo.
echo Features:
echo - Solo mode: translate to yourself
echo - Room mode: two-person conversation, joined by QR code
echo - VAD (voice activity detection)
echo - WebSocket transport in room mode
echo.

npx expo start --tunnel
