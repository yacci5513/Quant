#!/usr/bin/env bash
# Quant cron 등록 — 평일 KST 08:30 알림 + 09:30 자동 매매 + 16:00 마감 손익.
#
# 실행:
#   bash scripts/deploy/install-cron.sh
#
# 확인:
#   crontab -l
#   tail -f logs/cron/$(date +%Y-%m-%d).log
#
# 서버는 UTC. 모든 시각은 UTC로 환산해서 등록.
#   KST = UTC + 9
#   → KST 08:30 월~금 = UTC 23:30 일~목 (요일 -1 주의)
#   → KST 10:30 월~금 = UTC 01:30 월~금 (변동성 안정 + 추세 가드)
#   → KST 16:00 월~금 = UTC 07:00 월~금
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_DIR="$(pwd)"
DAILY_JOB="${PROJECT_DIR}/scripts/deploy/daily-job.sh"
AUTO_REBALANCE="${PROJECT_DIR}/scripts/deploy/auto-rebalance.sh"
EVENING_REPORT="${PROJECT_DIR}/scripts/deploy/evening-report.sh"
HOURLY_POSITIONS="${PROJECT_DIR}/scripts/deploy/hourly-positions.sh"

for f in "$DAILY_JOB" "$AUTO_REBALANCE" "$EVENING_REPORT" "$HOURLY_POSITIONS"; do
    [[ -x "$f" ]] || chmod +x "$f"
done

# 기존 quant 자동 cron 라인 제거 후 재등록 (일회성 cron은 보존)
TMP_CRON=$(mktemp)
trap 'rm -f "$TMP_CRON"' EXIT
crontab -l 2>/dev/null \
    | grep -v "daily-job.sh" \
    | grep -v "auto-rebalance.sh" \
    | grep -v "evening-report.sh" \
    | grep -v "hourly-positions.sh" \
    | grep -v "^# quant:" \
    > "$TMP_CRON" || true

{
    echo "# quant: daily signal (KST 08:30 월~금 = UTC 23:30 일~목)"
    echo "30 23 * * 0-4 ${DAILY_JOB}"
    echo "# quant: auto-rebalance (KST 10:30 월~금 = UTC 01:30 월~금) — 추세 가드용 1h 지연"
    echo "30 1 * * 1-5 ${AUTO_REBALANCE}"
    echo "# quant: hourly positions (KST 09~15 월~금 매시간 정각 = UTC 00~06)"
    echo "0 0-6 * * 1-5 ${HOURLY_POSITIONS}"
    echo "# quant: evening report (KST 16:00 월~금 = UTC 07:00 월~금)"
    echo "0 7 * * 1-5 ${EVENING_REPORT}"
} >> "$TMP_CRON"
crontab "$TMP_CRON"

echo "✅ cron 등록 완료:"
crontab -l
