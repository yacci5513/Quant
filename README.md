# Quant

한국투자증권 KIS Developers API 기반 KOSPI/KOSDAQ 자동매매 시스템.
백테스트 → 모의투자 → 소액 실거래 순으로 단계적 확장.

> 📖 **새 세션을 시작하셨다면 [`HANDOFF.md`](./HANDOFF.md)를 먼저 읽으세요.**
> 프로젝트 규칙·핸드오프 프로토콜은 [`CLAUDE.md`](./CLAUDE.md) 참조.

---

## Quick Start

```bash
# 1) 환경변수 템플릿 복사 후 채우기
cp .env.example .env
# (KIS 앱키/시크릿은 한투 비대면 계좌 개설 → KIS Developers에서 발급)

# 2) Docker 빌드 & 실행
docker compose up -d app

# 3) Jupyter 접속 (분석용)
docker compose up -d jupyter
open http://localhost:8888

# 4) 데이터 수집 (KOSPI 200 일봉 5년치)
docker compose exec app python -m quant.data.price.fetch_krx

# 5) 백테스트 실행 (모멘텀 Top-N 베이스라인)
docker compose exec app python -m quant.backtest.run --strategy momentum_topn

# 6) 컨테이너 셸 진입
docker compose exec app bash
```

---

## 기술 스택

- **Python 3.12** + **uv**
- **Docker** (Apple Silicon arm64)
- **데이터**: pykrx, FinanceDataReader → Parquet
- **저장**: SQLite (메타) + Parquet (시계열)
- **백테스트**: vectorbt
- **브로커**: KIS Developers (REST + WebSocket)
- **분석**: Jupyter

---

## 디렉토리 구조

```
Quant/
├── CLAUDE.md             # 프로젝트 규칙 + 핸드오프 프로토콜
├── HANDOFF.md            # 세션 인계 (항상 최신)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml        # uv
├── .env.example
├── src/quant/
│   ├── common/           # 설정, 로깅
│   ├── data/{price,news,disclosure}/
│   ├── nlp/              # 감성분석 (Phase 2)
│   ├── strategies/
│   ├── backtest/
│   └── live/             # KIS 클라이언트
├── data/                 # Parquet/SQLite (gitignored)
├── logs/
├── notebooks/            # 분석 노트북
└── tests/
```

---

## 진행 단계

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 가격 데이터 + 백테스트 + 모멘텀 베이스라인 | 진행 중 |
| 2 | OpenDART 공시 + 네이버 뉴스 수집 | 대기 |
| 3 | 가격-뉴스 상관관계 분석 | 대기 |
| 4 | 뉴스 시그널 결합 전략 백테스트 | 대기 |
| 5 | KIS 모의투자 → 소액 실거래 | 대기 |

상세 진행 상황은 [`HANDOFF.md`](./HANDOFF.md) 참조.

---

## 주의

- **모든 손익은 본인 책임**. 라이브러리/시스템 결함으로 인한 손실 책임지지 않음.
- **자본시장법 준수**: 시세조종성 주문(허수, 가장매매) 절대 금지. 미공개정보 이용 금지.
- **백테스트 ≠ 실거래 수익**: 슬리피지, 거래세, 호가 단위, 유동성 제약을 반드시 반영.
- **KIS API 호출 제한**: 실전 초당 20건, 모의 초당 2건. 토큰 일일 발급 한도 존재.
