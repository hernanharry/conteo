# Smoke check de la aplicacion (Fase 0).
# Verifica que la app arranca, SQLite inicializa, la camara responde y el
# stream MJPEG fluye. Uso (Windows):
#   powershell -File scripts/smoke_check.ps1 -BaseUrl http://localhost:8001 -Camera test-sintetica
param(
    [string]$BaseUrl = "http://localhost:8001",
    [string]$Camera = "test-sintetica"
)

$ErrorActionPreference = "Stop"

function Check($label, $url, [int]$ExpectedStatus = 200) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        $ok = ([int]$r.StatusCode -eq $ExpectedStatus)
        $detail = if ($r.Headers["Content-Type"]) { $r.Headers["Content-Type"] } else { "" }
        Write-Host ("{0,-45} {1}  [{2}] {3}" -f $label, $(if ($ok) { "OK" } else { "FAIL" }), $r.StatusCode, $detail)
        if (-not $ok) { return $false }
        return $true
    } catch {
        $msg = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { "no-http" }
        Write-Host ("{0,-45} FAIL  [{1}] {2}" -f $label, $msg, $_.Exception.Message)
        return $false
    }
}

$all = $true
$all = (Check "1. App arranca (/) ............." "$BaseUrl/") -and $all
$all = (Check "2. Galeria inicializada ........." "$BaseUrl/gallery") -and $all
$all = (Check "3. Pagina de vivo ..............." "$BaseUrl/live/$Camera") -and $all
$all = (Check "4. Stream MJPEG responde ......." "$BaseUrl/stream/$Camera") -and $all

try {
    $idx = (Invoke-WebRequest -Uri "$BaseUrl/" -UseBasicParsing -TimeoutSec 10).Content
    if ($idx -match "en vivo|en-vivo|reconectando|conectando") {
        Write-Host ("{0,-45} OK" -f "5. Estado de camara en tabla")
    } else {
        Write-Host ("{0,-45} FAIL (sin estado reconocible)" -f "5. Estado de camara en tabla")
        $all = $false
    }
} catch {
    Write-Host ("{0,-45} FAIL ({1})" -f "5. Estado de camara en tabla", $_.Exception.Message)
    $all = $false
}

if ($all) {
    Write-Host "`nSMOKE OK"
    exit 0
}
Write-Host "`nSMOKE FAIL"
exit 1