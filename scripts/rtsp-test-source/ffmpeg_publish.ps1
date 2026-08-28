# Publica un stream RTSP sintetico (patron de test + objeto en movimiento)
# hacia un servidor mediaMTX. Uso (Windows):
#   powershell -File scripts/rtsp-test-source/ffmpeg_publish.ps1 -Target rtsp://127.0.0.1:8554/feed1
param(
    [string]$Target = "rtsp://127.0.0.1:8554/feed1",
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Fps = 25
)

$ErrorActionPreference = "Stop"

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Error "ffmpeg no esta en el PATH. Instalelo (winget install ffmpeg) antes de continuar."
    exit 1
}

Write-Host "Publicando stream sintetico hacia $Target (Ctrl+C para detener)..."
Write-Host "Rectangle blanco se mueve en X para cruzar una linea de conteo."

& ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -f lavfi -i `
    "testsrc=size=${Width}x${Height}:rate=${Fps}" `
    -vf "drawbox=x=100+mod(100+t*${Width}/10\,${Width}):y=300:w=60:h=60:color=white@0.9:t=fill" `
    -c:v libx264 -preset ultrafast -tune zerolatency -crf 28 `
    -f rtsp -rtsp_transport tcp $Target

exit $LASTEXITCODE