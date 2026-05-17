# Quant — KOSPI Momentum + Market-Regime Filter

KOSPI 200 종목 대상 월간 모멘텀 + MA 시장 필터 백테스트·시그널 시스템. Python 3.12 + Docker.

> ⚠️ **투자 손실은 본인 책임입니다.** 본 코드는 백테스트·교육용이며 미래 수익을 보장하지 않습니다. 실거래 진입 전 [`QUANT_GUARDRAILS.md`](./QUANT_GUARDRAILS.md) 12개 체크리스트 통과 필수.

## 챔피언 전략

**월간 모멘텀 Top-10 + MA100 시장 필터**

```
시그널: 12-1 모멘텀 (Jegadeesh-Titman 1993)
시장:   KOSPI EW 평균 > 100일 이평선 일 때만 매수 (Faber 2007)
빈도:   월간 12회/년
```

5년 백테스트 (lookahead bias 자동 제거):
- 1년 평균 수익: +40.6% (14개 1년 윈도우 평균)
- 1년 최악: −8.1%, WIN: 86%
- Sharpe: 1.39

## Quick Start

```bash
# 환경 설정
cp .env.example .env
# .env에 본인 키 입력 (DART_API_KEY, TELEGRAM_BOT_TOKEN/CHAT_ID 등)

# Docker 빌드 + 데이터 수집
docker compose build app
docker compose run --rm app quant data fetch-krx --years 5

# 백테스트
docker compose run --rm app quant backtest momentum --top-n 10 --lookback 12

# 오늘 매수 시그널 (시드 1천만 기준)
docker compose run --rm app quant signal --daily --seed 10000000

# 텔레그램 자동 발송
docker compose run --rm app quant signal --daily --seed 10000000 --telegram
```

## 자동화 (한국 IP 서버 필요)

KIS API는 한국 IP만 허용. 한국 region 서버에 배포:

```bash
# 서버에서
git clone <repo> quant && cd quant
# .env는 별도 안전한 채널로 업로드 (예: scp, GitHub Secrets)
bash scripts/deploy/setup-server.sh
bash scripts/deploy/install-cron.sh   # 평일 18:30 KST 자동 실행
```

서버 정보, 인스턴스 IP, 사용자명 등 **민감 정보는 환경변수 / GitHub Secrets / 별도 사설 채널로 관리하고 코드에 절대 하드코딩하지 않습니다.**

## 디렉토리 구조

```
src/quant/
├── common/                # 설정, 로깅
├── data/
│   ├── price/             # KOSPI/KOSDAQ 일봉 (FinanceDataReader)
│   └── disclosure/        # OpenDART (자사주 공시, 재무제표)
├── strategies/            # 모멘텀, 저변동성, 신고가, 거래량 급증, 펀더멘털
├── backtest/              # 엔진, 시장 필터, 앙상블, 비용 모델, 메트릭
├── notify/                # 텔레그램 알림
├── daily_signal.py        # 매일 알림 (5가지 시나리오 자동 판별)
└── cli.py                 # quant CLI
```

## 핵심 모듈

| 모듈 | 역할 |
|---|---|
| `backtest/engine.py` | 가중치 기반 백테스트 + 자동 lookahead 보호 |
| `backtest/regime.py` | MA100/200 시장 레짐 필터 (Faber 2007) |
| `backtest/ensemble.py` | 다중 시그널 결합 |
| `strategies/momentum_topn.py` | 12-1 모멘텀 (챔피언) |
| `daily_signal.py` | 매일 알림 (정상/리밸런싱/청산/회복/Kill) |
| `notify/telegram.py` | 텔레그램 봇 발송 |

## 가드레일

[`QUANT_GUARDRAILS.md`](./QUANT_GUARDRAILS.md) 7대 함정 점검 필수:
1. 과최적화 · 2. 생존 편향 · 3. 미래 참조(엔진 자동 차단) · 4. 거래 비용 · 5. Crowding · 6. 데이터 품질 · 7. 통계적 유의성

라이브 진입 차단 룰:
- OOS Sharpe < 0.5 → 금지
- 백테스트 MDD > 30% → 금지 (시드 분할 후 가중 MDD < 30%면 듀얼 운용 허용)
- Kill switch 미구현 → 금지

## 보안

- `.env`는 절대 커밋 금지 (`.gitignore` + pre-commit hook 이중 차단)
- 시크릿 노출 시 30분 내 KIS 콘솔에서 키 무효화 + 재발급
- 서버 정보·인스턴스 ID·SSH 키는 GitHub Secrets / 환경변수에서만 참조

## 라이선스

MIT — `LICENSE` 참조. **No warranty, use at your own risk.**

## 학술 근거

- Jegadeesh & Titman (1993) — 모멘텀
- Faber (2007) — 시장 타이밍 (MA 필터)
- Frazzini & Pedersen (2014) — 저변동성
- Fama & French (1992) — 가치
- Ikenberry et al. (1995) — 자사주 매입
- Donchian (1960) / O'Neil — 신고가 돌파
