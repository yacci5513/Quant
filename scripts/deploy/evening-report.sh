#!/usr/bin/env bash
# 매일 평일 16:00 KST — 장 마감 후 손익 리포트.
#
# 흐름:
#   1. 데이터 fetch (오늘 종가 반영)
#   2. quant signal --daily --telegram (잔고 + 종목별 손익 알림)
#
# 매매는 안 함. 알림만.
#
# cron (UTC):
#   0 7 * * 1-5 = KST 16:00 월~금
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="logs/cron"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evening-$(date +%Y-%m-%d).log"

{
    echo
    echo "=========================================="
    echo "Evening report: $(date -Iseconds)"
    echo "=========================================="
    echo
    echo "[1/2] 종가 fetch"
    docker compose run --rm app quant data fetch-krx --years 1 2>&1 || \
        echo "⚠️ fetch-krx 실패 (계속 진행)"
    echo
    echo "[2/2] 잔고 + 손익 텔레그램 알림"
    docker compose run --rm app quant signal --daily --telegram 2>&1
    echo
    echo "Done: $(date -Iseconds)"
} >> "$LOG_FILE" 2>&1
