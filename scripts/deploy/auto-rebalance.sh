#!/usr/bin/env bash
# 매일 평일 10:30 KST — 자동 리밸런싱 (추세 가드용 1h 지연).
#
# 흐름:
#   1. 데이터 fetch (전일 종가까지)
#   2. quant live rebalance --execute
#      - 종목별 -10% 손절 / +20% 익절 (당일 추세 가드 포함)
#      - 포트 누적 -15% Kill switch (전량 매도)
#      - 시그널 NORMAL이면 위 가드 외 KIS 발주 0건
#      - REBALANCE/LIQUIDATE면 자동 매매
#
# 시그널 결과는 execute_rebalance가 텔레그램으로 직접 발송.
#
# cron (UTC):
#   30 1 * * 1-5 = KST 10:30 월~금
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="logs/cron"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/auto-rebalance-$(date +%Y-%m-%d).log"

{
    echo
    echo "=========================================="
    echo "Auto-rebalance started: $(date -Iseconds)"
    echo "=========================================="
    echo
    echo "[1/2] 최신 데이터 fetch"
    docker compose run --rm app quant data fetch-krx --years 1 2>&1 || \
        echo "⚠️ fetch-krx 실패 (계속 진행)"
    echo
    echo "[2/2] 자동 리밸런싱 (NORMAL이면 발주 0)"
    docker compose run --rm app quant live rebalance --execute 2>&1
    echo
    echo "Done: $(date -Iseconds)"
} >> "$LOG_FILE" 2>&1
