#!/usr/bin/env bash
# Quant 매일 자동 실행 — cron이 호출하는 진입점.
#
# 흐름:
#   1. 최신 가격 데이터 증분 fetch (오늘 종가)
#   2. 매일 알림 생성 → 텔레그램 발송
#
# 실행 시간 (한국 시간):
#   - 평일 08:30 권장 (장 시작 09:00 30분 전, 매수 결정 시간)
#   - 전일 종가 데이터 새벽에 갱신 완료된 상태 사용
#
# 로그:
#   - <project_root>/logs/cron/YYYY-MM-DD.log
set -euo pipefail

# 절대 경로로 이동 (cron은 PATH가 빈약함)
cd "$(dirname "$0")/../.."

LOG_DIR="logs/cron"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

# 로그 헤더
{
    echo
    echo "=========================================="
    echo "Daily job started: $(date -Iseconds)"
    echo "=========================================="
} >> "$LOG_FILE"

# 휴장일 체크 (토/일은 skip)
DOW=$(date +%u)  # 1=월, 7=일
if [[ "$DOW" -ge 6 ]]; then
    echo "Weekend ($DOW) — skip" >> "$LOG_FILE"
    exit 0
fi

# 1. 최신 데이터 fetch (실패해도 다음 단계 계속)
{
    echo
    echo "[1/2] Fetching latest prices..."
    docker compose run --rm app quant data fetch-krx --years 1 2>&1 || \
        echo "⚠️ fetch-krx failed (continuing)"
} >> "$LOG_FILE" 2>&1

# 2. 매일 알림 + 텔레그램
{
    echo
    echo "[2/2] Daily signal + telegram..."
    docker compose run --rm app quant signal --daily --seed 10000000 --telegram 2>&1
} >> "$LOG_FILE" 2>&1

echo "Daily job done: $(date -Iseconds)" >> "$LOG_FILE"
