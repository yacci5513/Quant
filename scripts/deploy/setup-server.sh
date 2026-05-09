#!/usr/bin/env bash
# Quant 서버 초기 셋업 (한국 region Linux 서버, Docker 사전 설치 가정).
#
# 실행 (서버에서):
#   git clone <repo-url> quant
#   cd quant
#   bash scripts/deploy/setup-server.sh
#
# 사전:
#   - Docker + Docker Compose 설치됨
#   - .env는 별도 안전한 채널로 업로드 (scp / GitHub Secrets / 사설 vault)
#   - 코드/문서에 서버 정보, 인스턴스 alias, IP 절대 하드코딩 금지
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== 1. .env 존재 확인 ==="
if [[ ! -f .env ]]; then
    echo "❌ .env 파일이 없습니다."
    echo "   별도 안전한 채널로 .env 업로드 후 재실행하세요."
    exit 1
fi
echo "✅ .env 존재 ($(wc -l < .env) 줄)"

echo
echo "=== 2. Docker 이미지 빌드 (5~10분) ==="
docker compose build app

echo
echo "=== 3. 데이터 디렉토리 준비 ==="
mkdir -p data/raw/prices data/raw/dart data/processed logs

echo
echo "=== 4. 첫 데이터 fetch (5년치, 5~10분) ==="
docker compose run --rm app quant data fetch-krx --years 5

echo
echo "=== 5. smoke test ==="
docker compose run --rm app quant hello
docker compose run --rm app pytest -q

echo
echo "=== 6. 텔레그램 ping (TELEGRAM_* 설정된 경우) ==="
docker compose run --rm app quant notify ping || echo "(텔레그램 미설정 — skip)"

echo
echo "✅ 셋업 완료. 다음:"
echo "   bash scripts/deploy/install-cron.sh"
