#!/usr/bin/env bash
# 장중 매시간 보유 손익 + 통합 잔고 알림 (매매 없음, 정보 전용).
#
# 발송: KIS 잔고 + 종목별 손익률·당일 수익률 + 통합 평가/매입/오늘·누적 손익.
# 시그널 계산 안 함 — 빠르고 가벼움.
#
# cron (UTC):
#   0 0-6 * * 1-5 = KST 09:00 ~ 15:00 월~금 매시간 정각 (7회/일)
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="logs/cron"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hourly-$(date +%Y-%m-%d).log"

{
    echo
    echo "=== Hourly positions: $(date -Iseconds) ==="
    docker compose run --rm app quant signal --positions --telegram 2>&1
} >> "$LOG_FILE" 2>&1
