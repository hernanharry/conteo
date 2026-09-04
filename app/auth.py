"""Autenticacion y autorizacion (F5) para la UI web, con Flask-Login.

Reemplaza la antigua HTTP Basic Auth (auth.py F5 original) por sesiones de
cookie gestionadas con Flask-Login y hashes de contrasena con Werkzeug.

Garantias:
- F5.1/F5.2: usuarios en la tabla `users`; contrasenas hasheadas con
  werkzeug.security.generate_password_hash (nunca en claro).
- F5.3: `require_login` (protege TODAS las rutas salvo login/health/static) y
  `require_admin` (solo role=admin: altas/bajas de camaras, borrado).
- F5.6: cookie de sesion siempre HttpOnly + SameSite=Lax; SESSION_COOKIE_SECURE
  se controla por variable de entorno (false en LAN/HTTP, true tras ngrok/HTTPS).
- LoginManager: el endpoint de login enrutado desde server.py llama a estos
  helpers (log_user_in / log_user_out).

El bootstrap del primer usuario admin se hace con scripts/create_admin.py
(F5.8), nunca con credenciales hardcodeadas.
"""

import os

import db
from flask_login import (
    LoginManager,
    UserMixin,
    login_user as _flask_login_user,
    logout_user as _flask_logout_user,
)
from werkzeug.security import check_password_hash, generate_password_hash

# F5.6: cookie segura segun entorno. En LAN se sirve por HTTP -> Secure=False.
# Tras un ngrok/Traefik en HTTPS, el operador la activa con SESSION_COOKIE_SECURE.
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# Rutas publicas SIEMPRE (sin login): el /health del orquestador (F7/F8) y el
# propio /login (para poder entrar). El resto de las rutas quedan protegidas.
PUBLIC_PATHS = (
    "/login",
    "/health",
    "/api/health",
    "/metrics",  # F7: Prometheus scrape puede ir sin auth (lectura de metricas)
)
# Prefijos publicos adicionales configurables (compatibilidad / flexibilidad).
_PUBLIC_PREFIXES = tuple(
    p for p in os.getenv("WEB_PUBLIC_PATHS", "").split(",") if p.strip()
)


def public_prefixes():
    return PUBLIC_PATHS + _PUBLIC_PREFIXES


def is_public(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Debe iniciar sesion para acceder."
login_manager.login_message_category = "warning"


class User(UserMixin):
    """Un usuario cargado desde la tabla `users` para la sesion."""

    def __init__(self, id_, username, role):
        self.id = id_
        self.username = username
        self.role = role

    def is_admin(self):
        return self.role == "admin"


@login_manager.user_loader
def _load_user(user_id):
    row = db.get_user_by_id(int(user_id))
    if row is None:
        return None
    return User(row["id"], row["username"], row["role"])


def authenticate_user(username, password):
    """Verifica credenciales contra la tabla users. Devuelve un objeto User si
    son validas, o None si no. Comparacion del hash en tiempo constante se
    garantiza dentro check_password_hash."""
    row = db.get_user(username)
    if row is None:
        return None
    if check_password_hash(row["password_hash"], password):
        return User(row["id"], row["username"], row["role"])
    return None


def hash_password(password):
    return generate_password_hash(password)


def log_user_in(user):
    _flask_login_user(user)


def log_user_out():
    _flask_logout_user()


def user_count():
    return len(db.list_users())
