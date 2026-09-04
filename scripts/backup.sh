#!/usr/bin/env bash
# backup.sh — F8.4: Backup automatizado de ./data (SQLite + galería).
#
# Uso:
#   ./scripts/backup.sh              # backup en ./data/backups/ con timestamp
#   ./scripts/backup.sh /ruta/out    # backup en directorio custom
#
# Crea un .tar.gz comprimido de todo ./data/ excluyendo los backups anteriores.
# Cron example (diario a las 3am):
#   0 3 * * * cd /ruta/al/proyecto && ./scripts/backup.sh >> ./data/backups/backup.log 2>&1
set -euo pipefail

DATA_DIR="${1:-./data}"
BACKUP_DIR="${DATA_DIR}/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${BACKUP_DIR}/conteo-backup-${TIMESTAMP}.tar.gz"
MAX_BACKUPS="${MAX_BACKUPS:-7}"

mkdir -p "${BACKUP_DIR}"

echo "[$(date -Iseconds)] Iniciando backup de ${DATA_DIR}..."

tar -czf "${BACKUP_FILE}" \
    --exclude="${BACKUP_DIR}" \
    -C "$(dirname "${DATA_DIR}")" \
    "$(basename "${DATA_DIR}")"

FILESIZE=$(du -sh "${BACKUP_FILE}" | cut -f1)
echo "[$(date -Iseconds)] Backup creado: ${BACKUP_FILE} (${FILESIZE})"

# Rotación: eliminar backups más antiguos que MAX_BACKUPS
BACKUP_COUNT=$(ls -1 "${BACKUP_DIR}"/conteo-backup-*.tar.gz 2>/dev/null | wc -l)
if [ "${BACKUP_COUNT}" -gt "${MAX_BACKUPS}" ]; then
    DELETE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    ls -1t "${BACKUP_DIR}"/conteo-backup-*.tar.gz | tail -n "${DELETE_COUNT}" | while read -r f; do
        echo "[$(date -Iseconds)] Eliminando backup antiguo: $(basename "$f")"
        rm -f "$f"
    done
fi

echo "[$(date -Iseconds)] Backup completado."
