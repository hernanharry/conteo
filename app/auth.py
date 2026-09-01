"""Autenticacion basica (F5) para la UI web y los streams.

Protege la interfaz de configuración (cámaras, vivo, galería, exports) y los
streams con HTTP Basic Auth. Las credenciales se leen de variables de entorno
`WEB_USER` / `WEB_PASSWORD`. Si no están configuradas, la autenticación queda
INACTIVA (comportamiento por defecto, compatible con instalaciones existentes
que no querían auth).

Garantias:
- Solo usa la biblioteca estandar (hmac.compare_digest, base64).
- Comparacion de contraseña en tiempo constante (hmac.compare_digest).
- No loguea credenciales.
- Protege todas las rutas EXCEPTO las de static y las de las listas blancas
  explicitas (para no romper img/embedding cuando se desea público).
"""

import base64
import hmac
import os

WEB_USER = os.getenv("WEB_USER")
WEB_PASSWORD = os.getenv("WEB_PASSWORD")

# Si no hay credenciales configuradas, la auth esta desactivada (None).
_credentials_configured = bool(WEB_USER and WEB_PASSWORD)

# Rutas que quedan PUBLICAS aunque haya auth (endpoints sin credenciales).
# Por defecto nada es publico si la auth esta activa, salvo que se liste aqui.
PUBLIC_PATHS = tuple(
    p for p in os.getenv("WEB_PUBLIC_PATHS", "").split(",") if p.strip()
)

AUTH_REALM = "object-tracker"
WWW_AUTHENTICATE = f'Basic realm="{AUTH_REALM}"'


def auth_enabled() -> bool:
    """True si la autenticacion esta activa (hay credenciales configuradas)."""
    return _credentials_configured


def _is_public(path: str) -> bool:
    """True si la ruta debe quedar fuera de la autenticacion (lista blanca)."""
    return any(path == p or path.startswith(p) for p in PUBLIC_PATHS)


def _check_credentials(username: str, password: str) -> bool:
    """Verifica usuario y password en tiempo constante (si esta configurado)."""
    if not hmac.compare_digest(str(username or ""), str(WEB_USER or "")):
        return False
    return hmac.compare_digest(str(password or ""), str(WEB_PASSWORD or ""))


def authenticate(request):
    """Valida una request con el header Authorization (Basic).

    Devuelve True si la request tiene credenciales validas (o si la auth esta
    desactivada). Devuelve False en caso contrario."""
    if not auth_enabled():
        return True
    if _is_public(request.path):
        return True

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(auth_header[6:]).decode("utf-8")
    except Exception:
        return False
    username, _, password = raw.partition(":")
    return _check_credentials(username, password)
