# ================================================================
# COMPETITION DAY - ngrok Static Domain Startup Script
# Stable URL that NEVER changes!
# ================================================================

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  COMPETITION DAY - ngrok Static Domain Startup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check ngrok
$ng = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ng) {
    Write-Host "ngrok not found. Please install it first!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Option 1: winget install ngrok.ngrok" -ForegroundColor Green
    Write-Host "Option 2: Download from https://ngrok.com/download" -ForegroundColor Green
    Write-Host ""
    Write-Host "After installing, run: ngrok config add-authtoken YOUR_TOKEN" -ForegroundColor Yellow
    Write-Host "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] ngrok found." -ForegroundColor Green

# ===========================================
# IMPORTANT: Set your free static domain here!
# Get it from: https://dashboard.ngrok.com/domains
# ===========================================
$NGROK_STATIC_DOMAIN = $env:NGROK_STATIC_DOMAIN
if (-not $NGROK_STATIC_DOMAIN) {
    $NGROK_STATIC_DOMAIN = "pixilated-spraylike-zona.ngrok-free.dev"
}

Write-Host "[OK] Static domain: $NGROK_STATIC_DOMAIN" -ForegroundColor Green

# Step 2: Cleanup old server processes
Write-Host "Cleaning up old processes..." -ForegroundColor Yellow
try {
    Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "ngrok" -Force -ErrorAction SilentlyContinue
} catch {}
Start-Sleep -Seconds 2

# Step 3: Start Server
Write-Host "Starting local server on port 8001..." -ForegroundColor Yellow
$serverProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.main:app --host 0.0.0.0 --port 8001" -WorkingDirectory "c:\work\hcl" -PassThru -NoNewWindow
Start-Sleep -Seconds 5

# Step 4: Start ngrok with static domain
Write-Host ""
Write-Host "Starting ngrok with static domain..." -ForegroundColor Yellow
Write-Host "(This URL NEVER changes!)" -ForegroundColor DarkGray

$ngrokProcess = Start-Process -FilePath "ngrok" -ArgumentList "http 8001 --domain=$NGROK_STATIC_DOMAIN" -PassThru -NoNewWindow
Start-Sleep -Seconds 5

# Step 5: Display info
$fullUrl = "https://$NGROK_STATIC_DOMAIN"

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  SYSTEM READY!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Your PERMANENT URL:" -ForegroundColor White
Write-Host "  $fullUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "  API Endpoint:" -ForegroundColor White
Write-Host "  $fullUrl/api/scam-honeypot" -ForegroundColor Cyan
Write-Host ""
Write-Host "  This URL NEVER changes! Set it ONCE in HF Secrets." -ForegroundColor Green
Write-Host ""
Write-Host "  HF Space Secret:" -ForegroundColor Yellow
Write-Host "  LOCAL_TUNNEL_URL = $fullUrl" -ForegroundColor White
Write-Host ""

# Copy to clipboard
try {
    Set-Clipboard -Value $fullUrl
    Write-Host "  (Copied to clipboard!)" -ForegroundColor Green
} catch {}

Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host "================================================" -ForegroundColor Green

# Keep running until Ctrl+C
try {
    while ($true) { Start-Sleep -Seconds 60 }
} finally {
    Write-Host "Shutting down..." -ForegroundColor Yellow
    Stop-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $ngrokProcess.Id -ErrorAction SilentlyContinue
}
