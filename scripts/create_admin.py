"""Bootstrap del primer usuario admin (F5.8).

Crea el usuario administrador inicial. NUNCA usa credenciales hardcodeadas:
las pide por entrada interactiva o por argumentos/entorno explicitos.

Uso:
    python scripts/create_admin.py                     # prompter username + password
    python scripts/create_admin.py --username admin    # prompter solo password
    ADMIN_USER=admin ADMIN_PASSWORD=... python scripts/create_admin.py --non-interactive

Obligatorio:
    DB_PATH   (se hereda de .env si se corre con el contenedor / compose env_file)

Requisitos de seguridad:
    - Si el usuario ya existe, se informa y NO se toca su password (salvo
      --force-password).
    - El password jamas se loguea ni se escribe en BD; solo su hash.
"""
import argparse
import getpass
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, "app")
sys.path.insert(0, APP_DIR)

import db  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Crea el primer usuario admin (F5.8).")
    ap.add_argument("--username", default=None, help="usuario (si omite, se pide)")
    ap.add_argument("--role", default="admin", choices=["admin", "viewer"], help="rol por defecto admin")
    ap.add_argument("--force-password", action="store_true",
                    help="reinicia el password aunque el usuario ya exista")
    ap.add_argument("--non-interactive", action="store_true",
                    help="lee ADMIN_USER/ADMIN_PASSWORD del entorno (no chamuyo)")
    args = ap.parse_args()

    db.init_db()

    interactive = not args.non_interactive

    if args.username:
        username = args.username.strip()
    elif interactive:
        username = input("Usuario: ").strip()
    else:
        username = (os.getenv("ADMIN_USER") or "").strip()

    if not username:
        print("error: falta el nombre de usuario", file=sys.stderr)
        sys.exit(1)

    if args.non_interactive:
        password = os.getenv("ADMIN_PASSWORD")
        if not password:
            print("error: ADMIN_PASSWORD no esta definido", file=sys.stderr)
            sys.exit(1)
    else:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Repetir password: ")
        if password != confirm:
            print("error: los passwords no coinciden", file=sys.stderr)
            sys.exit(1)

    if not password:
        print("error: el password no puede estar vacio", file=sys.stderr)
        sys.exit(1)

    import auth  # noqa: E402

    existing = db.get_user(username)
    if existing:
        if not args.force_password:
            print(f"el usuario '{username}' ya existe (rol={existing['role']}); no se modifico nada.")
            print("use --force-password para reiniciar el password.")
            return
        db.update_user_role(username, args.role) if False else None
        # reinicia el password: actualizamos el hash.
        import sqlite3  # noqa: E402

        conn = db.get_conn()
        try:
            conn.execute("UPDATE users SET password_hash=? WHERE username=?",
                         (auth.hash_password(password), username))
            conn.commit()
        finally:
            conn.close()
        print(f"password de '{username}' reiniciado (rol {args.role}).")
        return

    db.create_user(username, auth.hash_password(password), role=args.role)
    if len(db.list_users()) == 1:
        print(f"primer usuario admin '{username}' creado.")
    else:
        print(f"usuario '{username}' (rol={args.role}) creado.")


if __name__ == "__main__":
    main()
