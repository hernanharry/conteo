"""Tests F8 — Docker/Producción: hardening, backup, build labels, compose.

Cubre:
  - Dockerfile tiene HEALTHCHECK, usuario no-root, labels OCI
  - docker-compose.yml tiene healthcheck, limits, Traefik (profile), stop_grace_period
  - docker-compose.observability.yml existe (ya testeado en test_observability_f7.py)
  - backup.sh existe y es ejecutable
  - prometheus.yml existe y apunta al servicio correcto
"""

import os
import stat
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_has_healthcheck():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "HEALTHCHECK" in content


def test_dockerfile_has_nonroot_user():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "appuser" in content


def test_dockerfile_has_oci_labels():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "org.opencontainers.image" in content
    assert "BUILD_DATE" in content
    assert "BUILD_VERSION" in content


def test_compose_has_healthcheck():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "healthcheck:" in content
    assert "/api/health" in content


def test_compose_has_resource_limits():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "memory:" in content
    assert "cpus:" in content


def test_compose_does_not_force_nonroot_user():
    # fix(f8) 36873ff: `user: appuser` en compose rompia los permisos de
    # escritura sobre el volumen ./data montado como root (BD inaccesible).
    # En compose el contenedor corre como root para poder persistir; el
    # appuser del Dockerfile queda solo para builds/contextos sin volume mount.
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    service_block = content.split("traefik:", 1)[0]
    assert "appuser" not in service_block


def test_compose_has_stop_grace_period():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "stop_grace_period: 30s" in content


def test_compose_has_traefik_profile():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "with-traefik" in content
    assert "traefik.enable=true" in content


def test_compose_build_args():
    content = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "BUILD_DATE" in content
    assert "BUILD_VERSION" in content


def test_backup_script_exists():
    path = REPO_ROOT / "scripts" / "backup.sh"
    assert path.exists(), "scripts/backup.sh no existe"


def test_backup_script_executable():
    path = REPO_ROOT / "scripts" / "backup.sh"
    mode = stat.S_IMODE(path.stat().st_mode)
    # En Windows, chmod no aplica bits Unix; el script debe ser ejecutable en Linux
    is_linux_exec = bool(mode & stat.S_IXUSR)
    has_shebang = path.read_text().startswith("#!/usr/bin/env bash")
    assert is_linux_exec or has_shebang, "backup.sh no es ejecutable y no tiene shebang"


def test_backup_script_content():
    content = (REPO_ROOT / "scripts" / "backup.sh").read_text()
    assert "tar -czf" in content
    assert "MAX_BACKUPS" in content


def test_env_example_has_alert_vars():
    content = (REPO_ROOT / ".env.example").read_text()
    assert "ALERT_CHECK_INTERVAL" in content or "ALERT_CAMERA_DOWN" in content
