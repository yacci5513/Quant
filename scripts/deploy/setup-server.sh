#!/usr/bin/env bash
# Quant 서버 초기 셋업 (AWS Lightsail / Ubuntu).
#
# 실행:
#   ssh ***SERVER***
#   cd ~ && git clone git@github.com:***USER***/Quant.git quant
#   cd quant
#   bash scripts/deploy/setup-server.sh
#
# 사전:
#   - Docker + Compose 설치됨 (이미 ***SERVER***에 있음)
#   - .env는 별도 업로드 (scp .env ***SERVER***:~/quant/.env)
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== 1. .env 존재 확인 ==="
if [[ ! -f .env ]]; then
    echo "❌ .env 파일이 없습니다."
    echo "   로컬에서 다음 명령으로 업로드:"
    echo "   scp .env ***SERVER***:$(pwd)/.env"
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
echo "=== 6. 텔레그램 ping ==="
docker compose run --rm app quant notify ping

echo
echo "✅ 셋업 완료. 다음:"
echo "   bash scripts/deploy/install-cron.sh"
