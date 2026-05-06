"""백테스트 엔진 (vectorbt 래퍼).

가격 패널 + 포지션 가중치(weights)를 받아 포트폴리오 시뮬레이션을 수행한다.
가드레일 자동 검증 후 메트릭을 반환.

설계:
- 시그널은 호출자가 미리 계산해서 전달 (`shift(1)`을 호출자가 책임)
- weights[t] = 0~1 사이 비율, 종목별 합 ≤ 1.0
- 비용은 CostModel로 통합 주입
- vectorbt가 기본 인프라(체결가, 시뮬레이션, 회전율 계산) 처리
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS, CostModel
from quant.backtest.metrics import Metrics, compute_metrics
from quant.common.logger import logger


@dataclass
class BacktestResult:
    """백테스트 단일 실행 결과."""

    equity_curve: pd.Series  # 자본 곡선 (1.0 시작)
    daily_returns: pd.Series  # 일별 수익률
    metrics: Metrics
    cost_model: CostModel
    n_rebalances: int  # 리밸런싱 횟수
    avg_n_holdings: float  # 평균 보유 종목 수


def _validate_inputs(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    """가드레일 §3 (룩어헤드) 입력 검증."""
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices 인덱스가 정렬되지 않음")
    if not weights.index.is_monotonic_increasing:
        raise ValueError("weights 인덱스가 정렬되지 않음")
    common_cols = set(prices.columns) & set(weights.columns)
    if not common_cols:
        raise ValueError("prices와 weights에 공통 종목이 없음")
    # weights 합이 1.0 초과인 경우 — 실수로 정규화 누락
    row_sums = weights.fillna(0).sum(axis=1)
    if (row_sums > 1.0001).any():
        bad = weights.index[row_sums > 1.0001][0]
        logger.warning(f"weights 합 > 1.0 발견 ({bad}): {row_sums.loc[bad]:.4f} — 정규화 누락 의심")


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    cost_model: CostModel = BLUECHIP_KIS,
    init_cash: float = 10_000_000.0,
    auto_shift_weights: bool = True,
) -> BacktestResult:
    """가중치 기반 백테스트.

    prices: 일별 종가 패널 (index=date, columns=ticker)
    weights: 일별 목표 가중치 (0~1, 합 ≤ 1.0). 리밸런싱 일자에만 값,
             나머지는 NaN 또는 직전 값 forward-fill 가능.
    auto_shift_weights: True면 weights를 한 영업일 .shift(1)해서 적용.
        해석: "t 시점 종가로 계산한 시그널은 t+1에 매수 → t+1 수익부터 반영".
        가드레일 §3 룩어헤드 자동 보호. False면 기존 동작 (호출자 책임).

    가드레일:
    - 매일 빈도 + breakout 결합 시 발견된 lookahead 누출 자동 차단
    - 격주 이하 빈도엔 영향 미미 (가중치 변경 드물어 +0~0.5% 차이)
    - prices/weights 인덱스 정렬 검증
    - 가중치 합 1.0 초과 경고
    """
    _validate_inputs(prices, weights)

    # 공통 종목·일자만 사용
    common_cols = sorted(set(prices.columns) & set(weights.columns))
    common_idx = prices.index.intersection(weights.index)
    px = prices.loc[common_idx, common_cols].astype(float)
    w = weights.loc[common_idx, common_cols].astype(float).fillna(0.0)

    # 가드레일 §3: 룩어헤드 자동 보호 — t 종가 시그널을 t+1 매수로 해석
    if auto_shift_weights:
        w = w.shift(1).fillna(0.0)

    # 일별 종목 수익률
    rets = px.pct_change().fillna(0.0)

    # 포트폴리오 일별 수익률: 전일 가중치 × 당일 수익률 (룩어헤드 안전)
    # weights[t]는 t 시작 시점 보유 비중으로 해석 → t 수익률에 그대로 적용
    port_rets_gross = (w * rets).sum(axis=1)

    # 회전율 = 가중치 변화량 합 / 2 (단방향)
    weight_change = w.diff().abs().fillna(w.iloc[0].abs())
    turnover_per_day = weight_change.sum(axis=1) / 2.0

    # 비용: 회전율 × 한쪽당 평균 비용
    cost_per_day = turnover_per_day * cost_model.avg_fee_per_side
    port_rets_net = port_rets_gross - cost_per_day

    # 자본 곡선
    equity = (1.0 + port_rets_net).cumprod()

    # 리밸런싱/통계
    rebalances = int((turnover_per_day > 1e-6).sum())
    avg_holdings = float((w > 1e-6).sum(axis=1).mean())
    turnover_annual_avg = float(
        turnover_per_day.sum() / max((px.index[-1] - px.index[0]).days / 365.25, 1e-9)
    )

    # 거래(매수 신규 진입) 카운트 — 새로 진입한 (종목,날짜) 페어
    entries = (w > 0) & (w.shift(1).fillna(0) == 0)
    n_trades = int(entries.sum().sum())

    # 승률: 거래별 손익 (간단 추정 — 진입 후 청산 시 종목 수익률)
    win_rate = _estimate_win_rate(w, px)

    metrics = compute_metrics(
        port_rets_net,
        n_trades=n_trades,
        win_rate=win_rate,
        turnover_annual=turnover_annual_avg,
    )

    return BacktestResult(
        equity_curve=equity,
        daily_returns=port_rets_net,
        metrics=metrics,
        cost_model=cost_model,
        n_rebalances=rebalances,
        avg_n_holdings=avg_holdings,
    )


def _estimate_win_rate(w: pd.DataFrame, px: pd.DataFrame) -> float | None:
    """간단 승률 추정 — 보유 구간 단위로 종목별 수익률 부호.

    완벽하진 않지만 트렌드 가늠용. 정밀하게는 vbt.Portfolio.trades 사용 권장.
    """
    held = w > 0
    if not held.any().any():
        return None
    wins = 0
    total = 0
    for col in w.columns:
        h = held[col].astype(int)
        # 진입(0→1)과 청산(1→0) 인덱스
        diff = h.diff().fillna(h.iloc[0])
        starts = h.index[diff == 1].tolist()
        ends = h.index[diff == -1].tolist()
        if h.iloc[-1] == 1:
            ends.append(h.index[-1])
        for s, e in zip(starts, ends, strict=False):
            if s == e:
                continue
            ret = px.loc[e, col] / px.loc[s, col] - 1.0
            if pd.isna(ret):
                continue
            total += 1
            if ret > 0:
                wins += 1
    if total == 0:
        return None
    return wins / total


def run_split(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    is_ratio: float = 0.7,
    cost_model: CostModel = BLUECHIP_KIS,
) -> tuple[BacktestResult, BacktestResult]:
    """IS/OOS 시간 분리 후 각각 백테스트.

    핵심: weights는 호출자가 전체 기간을 이미 계산했어야 함 (전략 자체는 IS만 보고
    파라미터 결정 후, 동일 룰로 OOS에도 적용). 본 함수는 결과 섹션만 분리.
    """
    n = len(prices.index)
    cut = int(n * is_ratio)
    is_idx = prices.index[:cut]
    oos_idx = prices.index[cut:]

    is_res = run_backtest(prices.loc[is_idx], weights.loc[is_idx], cost_model=cost_model)
    oos_res = run_backtest(prices.loc[oos_idx], weights.loc[oos_idx], cost_model=cost_model)
    return is_res, oos_res


def _safe_divide(num: float, den: float) -> float:
    if den == 0 or np.isnan(den):
        return 0.0
    return num / den
