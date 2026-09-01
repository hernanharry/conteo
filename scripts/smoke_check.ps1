# Smoke check de la aplicacion (Fase 0 + F5/F7).
# Verifica que la app arranca, SQLite inicializa, la camara responde, el
# health check responde y el stream MJPEG fluye. Uso (Windows):
#   powershell -File scripts/smoke_check.ps1 -BaseUrl http://localhost:8001 -Camera test-sintetica
# Si la app tiene auth habilitada (WEB_USER/WEB_PASSWORD en .env, F5), pasar
# tambien -User y -Password:
#   powershell -File scripts/smoke_check.ps1 -BaseUrl http://localhost:8001 -Camera test-sintetica -User admin -Password "secreto"
param(
    [string]$BaseUrl = "http://localhost:8001",
    [string]$Camera = "test-sintetica",
    [string]$User = "",
    [string]$Password = ""
)

$ErrorActionPreference = "Stop"

# F5: si se pasan credenciales, arma el header Authorization (Basic).
$headers = @{}
if ($User -and $Password) {
    $plain = "$User`:$Password"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($plain)
    $token = [Convert]::ToBase64String($bytes)
    $headers["Authorization"] = "Basic $token"
}

function Check($label, $url, [int]$ExpectedStatus = 200) {
    try {
        $r = Invoke-WebRequest -Uri $url -Headers $headers -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
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

# F6: el stream MJPEG es infinito por diseno; Invoke-WebRequest queda
# bufferizando hasta el cierre o el timeout. Se lee solo el primer fragmento
# del body (boundary + JPEG) y se cierra la conexion.
function CheckStream($label, $url) {
    try {
        $req = [System.Net.HttpWebRequest]::Create($url)
        $req.Timeout = 10000
        $req.ReadWriteTimeout = 10000
        $req.AllowAutoRedirect = $false
        foreach ($k in $headers.Keys) { $req.Headers[$k] = $headers[$k] }
        $resp = $req.GetResponse()
        $bytes = [byte[]]::new(8192)
        $stream = $resp.GetResponseStream()
        $n = $stream.Read($bytes, 0, $bytes.Length)
        $resp.Close()
        $ct = [string]$resp.ContentType
        # Busca el SOI de un JPEG (FF D8) en el primer fragmento del body.
        $jpeg = $false
        for ($i = 0; $i -lt $n - 1; $i++) {
            if ($bytes[$i] -eq 0xFF -and $bytes[$i + 1] -eq 0xD8) { $jpeg = $true; break }
        }
        $ok = ($ct -match "multipart/x-mixed-replace" -and $jpeg)
        $detail = if ($ok) { "$ct, jpg+$n-bytes" } else { if ($n -eq 0) { "empty" } else { "sin SOI" } }
        Write-Host ("{0,-45} {1}  [{2}]" -f $label, $(if ($ok) { "OK" } else { "FAIL" }), $detail)
        return $ok
    } catch {
        Write-Host ("{0,-45} FAIL  [{1}]" -f $label, $_.Exception.Message)
        return $false
    }
}

$all = $true
# F7: el health check debe responder 200 (es publico aun con auth).
$all = (Check "1. Health check (/api/health) ..." "$BaseUrl/api/health") -and $all
$all = (Check "2. App arranca (/) ............." "$BaseUrl/") -and $all
$all = (Check "3. Galeria inicializada ........." "$BaseUrl/gallery") -and $all
$all = (Check "4. Pagina de vivo ..............." "$BaseUrl/live/$Camera") -and $all
$all = (CheckStream "5. Stream MJPEG responde ......." "$BaseUrl/stream/$Camera") -and $all

try {
    $idx = (Invoke-WebRequest -Uri "$BaseUrl/" -Headers $headers -UseBasicParsing -TimeoutSec 10).Content
    if ($idx -match "en vivo|en-vivo|reconectando|conectando") {
        Write-Host ("{0,-45} OK" -f "6. Estado de camara en tabla")
    } else {
        Write-Host ("{0,-45} FAIL (sin estado reconocible)" -f "6. Estado de camara en tabla")
        $all = $false
    }
} catch {
    Write-Host ("{0,-45} FAIL ({1})" -f "6. Estado de camara en tabla", $_.Exception.Message)
    $all = $false
}

if ($all) {
    Write-Host "`nSMOKE OK"
    exit 0
}
Write-Host "`nSMOKE FAIL"
exit 1