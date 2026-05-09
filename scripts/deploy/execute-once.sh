#!/usr/bin/env bash
# 1회용 자동 매매 실행 (예: 챔피언 룰 외 시작점 첫 매수).
#
# 사용:
#   bash scripts/deploy/execute-once.sh         # dry-run (시뮬, 텔레그램만)
#   bash scripts/deploy/execute-once.sh execute # 실거래 주문
#
# 일정: cron 또는 직접 실행. 사용 후 cron에서 제거하는 게 원칙.
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="logs/cron"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/execute-once-$(date +%Y-%m-%d).log"

MODE="${1:-dry-run}"
if [[ "$MODE" == "execute" ]]; then
    FLAG="--execute"
else
    FLAG="--dry-run"
fi

{
    echo
    echo "=========================================="
    echo "Execute-once started: $(date -Iseconds)"
    echo "Mode: $MODE"
    echo "=========================================="
    echo
    echo "[1/2] 최신 데이터 fetch"
    docker compose run --rm app quant data fetch-krx --years 1 2>&1 || \
        echo "⚠️ fetch-krx 실패 (계속 진행)"
    echo
    echo "[2/2] 자동 리밸런싱 ($FLAG)"
    docker compose run --rm app quant live rebalance "$FLAG" 2>&1
    echo
    echo "Done: $(date -Iseconds)"
} >> "$LOG_FILE" 2>&1

echo "✅ 완료. 로그: $LOG_FILE"
