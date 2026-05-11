#!/usr/bin/env bash
# Quant cron 등록 — 평일 08:30 KST 매일 알림 (장 시작 30분 전).
#
# 실행:
#   bash scripts/deploy/install-cron.sh
#
# 확인:
#   crontab -l                # 현재 등록된 cron 보기
#   tail -f logs/cron/$(date +%Y-%m-%d).log   # 오늘 실행 로그
set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_DIR="$(pwd)"
DAILY_JOB="${PROJECT_DIR}/scripts/deploy/daily-job.sh"

if [[ ! -x "$DAILY_JOB" ]]; then
    chmod +x "$DAILY_JOB"
fi

# cron 라인 — KST 08:30 월~금 (장 시작 09:00 30분 전, 매수 결정 시간)
# 서버는 UTC. KST = UTC+9 → KST 08:30 월~금 = UTC 23:30 일~목 (요일 -1 주의)
CRON_LINE="30 23 * * 0-4 ${DAILY_JOB}"
COMMENT="# quant: daily signal + telegram (KST 08:30 월~금 = UTC 23:30 일~목)"

# 기존 cron에 quant 라인 있으면 제거 후 재등록
TMP_CRON=$(mktemp)
trap 'rm -f "$TMP_CRON"' EXIT
crontab -l 2>/dev/null | grep -v "quant.*daily" | grep -v "$DAILY_JOB" > "$TMP_CRON" || true
{
    echo "$COMMENT"
    echo "$CRON_LINE"
} >> "$TMP_CRON"
crontab "$TMP_CRON"

echo "✅ cron 등록 완료:"
crontab -l | grep -A 1 "quant" || crontab -l | tail -3

echo
echo "테스트 실행 (지금 한 번 돌려보기):"
echo "  bash $DAILY_JOB"
echo
echo "로그 확인:"
echo "  tail -f $PROJECT_DIR/logs/cron/\$(date +%Y-%m-%d).log"
