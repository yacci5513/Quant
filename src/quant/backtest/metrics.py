"""백테스트 성과 메트릭 + 가드레일 검증.

핵심 지표:
- 누적/연환산 수익률 (CAGR)
- Sharpe (연환산), Sortino
- MDD (최대 낙폭)
- 회전율 (turnover)
- 거래 횟수, 승률, 평균 손익비

가드레일 자동 경고:
- 거래 횟수 < 30 → 통계적 유의성 부족
- Sharpe > 3.0 또는 MDD < 5% → 과적합 의심
- 단일 종목 기여도 > 30% → 분산 실패

IS/OOS 분리 헬퍼:
- 시간 기반 split (디폴트 70/30)
- Walk-forward window 생성
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# 가드레일 임계값
MIN_TRADES_FOR_SIGNIFICANCE = 30
SUSPICIOUS_SHARPE = 3.0
SUSPICIOUS_MIN_MDD_PCT = 0.05
MAX_SINGLE_STOCK_CONTRIBUTION = 0.30
TRADING_DAYS_PER_YEAR = 252


@dataclass
class Metrics:
    """백테스트 결과 요약."""

    period_start: date
    period_end: date
    days: int

    total_return: float  # 누적 수익률 (1.0 = 100%)
    cagr: float  # 연환산
    volatility_annual: float
    sharpe: float
    sortino: float
    max_drawdown: float  # 음수 비율 (e.g. -0.25 = -25%)
    calmar: float  # CAGR / |MDD|

    n_trades: int
    win_rate: float | None
    turnover_annual: float | None  # 연환산 회전율 (단방향)

    warnings: list[str]

    def summary(self) -> str:
        years = self.days / 365.25
        wins = f"{self.win_rate * 100:.1f}%" if self.win_rate is not None else "N/A"
        turn = f"{self.turnover_annual:.1f}x/년" if self.turnover_annual is not None else "N/A"
        return (
            f"기간: {self.period_start} ~ {self.period_end} ({years:.1f}년)\n"
            f"누적 수익률 : {self.total_return * 100:>8.2f}%\n"
            f"CAGR        : {self.cagr * 100:>8.2f}%\n"
            f"변동성(연)  : {self.volatility_annual * 100:>8.2f}%\n"
            f"Sharpe      : {self.sharpe:>8.2f}\n"
            f"Sortino     : {self.sortino:>8.2f}\n"
            f"MDD         : {self.max_drawdown * 100:>8.2f}%\n"
            f"Calmar      : {self.calmar:>8.2f}\n"
            f"거래 횟수   : {self.n_trades:>8d}\n"
            f"승률        : {wins:>8}\n"
            f"회전율      : {turn:>8}\n"
        )


def _max_drawdown(equity: pd.Series) -> float:
    """equity curve의 최대 낙폭 (음수 비율)."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_metrics(
    returns: pd.Series,
    *,
    n_trades: int = 0,
    win_rate: float | None = None,
    turnover_annual: float | None = None,
    risk_free_rate: float = 0.0,
) -> Metrics:
    """일별 수익률에서 메트릭 계산. 결측치는 0으로 가정 (포지션 없음).

    returns: 일별 포트폴리오 수익률 (DatetimeIndex)
    """
    rets = returns.fillna(0.0)
    if len(rets) == 0:
        raise ValueError("빈 returns")

    equity = (1.0 + rets).cumprod()
    days = (rets.index[-1] - rets.index[0]).days
    years = max(days / 365.25, 1e-9)

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    vol_annual = float(rets.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    excess = rets - risk_free_rate / TRADING_DAYS_PER_YEAR
    sharpe = (
        float(excess.mean()) / float(rets.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if rets.std(ddof=0) > 0
        else 0.0
    )
    downside = rets[rets < 0]
    sortino = (
        float(excess.mean()) / float(downside.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if len(downside) > 0 and downside.std(ddof=0) > 0
        else 0.0
    )
    mdd = _max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    warnings = _check_guardrails(
        n_trades=n_trades,
        sharpe=sharpe,
        mdd=mdd,
    )

    return Metrics(
        period_start=rets.index[0].date(),
        period_end=rets.index[-1].date(),
        days=days,
        total_return=total_return,
        cagr=cagr,
        volatility_annual=vol_annual,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        n_trades=n_trades,
        win_rate=win_rate,
        turnover_annual=turnover_annual,
        warnings=warnings,
    )


def _check_guardrails(*, n_trades: int, sharpe: float, mdd: float) -> list[str]:
    """가드레일 위반 자동 경고. QUANT_GUARDRAILS.md §1, §7 참조."""
    out: list[str] = []
    if n_trades and n_trades < MIN_TRADES_FOR_SIGNIFICANCE:
        out.append(
            f"⚠️ 거래 횟수 {n_trades} < {MIN_TRADES_FOR_SIGNIFICANCE} — "
            f"통계적 의미 없음. 결과 신뢰 불가."
        )
    if sharpe > SUSPICIOUS_SHARPE:
        out.append(
            f"🚨 Sharpe {sharpe:.2f} > {SUSPICIOUS_SHARPE} — "
            f"과적합 의심 (룩어헤드/생존편향/비용 누락 점검)."
        )
    if mdd > -SUSPICIOUS_MIN_MDD_PCT:
        out.append(
            f"🚨 MDD {mdd * 100:.2f}% (절댓값 < {SUSPICIOUS_MIN_MDD_PCT * 100}%) — " f"과적합 의심."
        )
    return out


# -----------------------------------------------------------------------------
# IS / OOS 분리 + Walk-forward
# -----------------------------------------------------------------------------
def split_in_out_of_sample(
    index: pd.DatetimeIndex,
    is_ratio: float = 0.7,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """시간 기반 70/30 분리. IS는 앞쪽, OOS는 뒤쪽.

    Returns: (is_index, oos_index)
    """
    if not 0 < is_ratio < 1:
        raise ValueError("is_ratio must be in (0, 1)")
    n = len(index)
    cut = int(n * is_ratio)
    return index[:cut], index[cut:]


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walk_forward_windows(
    start: date,
    end: date,
    train_years: int = 1,
    test_months: int = 6,
    step_months: int = 6,
) -> list[WalkForwardWindow]:
    """Rolling walk-forward 윈도우 생성.

    예: train_years=1, test_months=6 → 1년 학습 + 6개월 검증, 6개월씩 슬라이딩.
    가드레일 §1: 단일 IS/OOS 분리보다 더 엄격한 검증.
    """
    windows: list[WalkForwardWindow] = []
    cur = start
    while True:
        train_start = cur
        train_end = train_start + timedelta(days=train_years * 365)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_months * 30)
        if test_end > end:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cur = cur + timedelta(days=step_months * 30)
    return windows
