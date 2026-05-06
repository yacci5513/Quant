"""벤치마크 — 전략 유니버스의 동일가중 평균.

용도:
- 전략 vs 단순 분산투자 비교
- "베이스라인을 못 이기면 의미 없다" 가드레일 §1 검증

Equal-weighted KOSPI 200 ≈ 매일 200종목 1/200씩 보유.
실제 KOSPI 200 ETF(069500)는 시총 가중이라 다르지만, 본 전략이 동일 가중이므로
공정 비교를 위해 동일 가중 벤치마크 사용.
"""

from __future__ import annotations

import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS, CostModel
from quant.backtest.engine import BacktestResult, run_backtest


def equal_weight_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """매일 모든 종목 1/N씩 보유하는 가중치 패널.

    데이터 시작일에는 일괄 진입(전체 회전율 1.0). 이후엔 가격 변동에 따라
    실효 가중치가 약간 흔들리지만 본 함수는 단순화를 위해 항상 1/N로 고정.
    """
    n = prices.shape[1]
    if n == 0:
        return prices.copy()
    w = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    # 가격이 NaN인 칸은 0 (해당 종목 미상장 또는 거래정지)
    w = w.where(prices.notna(), 0.0)
    # 매일 전체 합이 1.0이 되도록 정규화 (NaN 제외)
    row_sums = w.sum(axis=1)
    w = w.div(row_sums.where(row_sums > 0, 1), axis=0)
    return w


def run_benchmark(
    prices: pd.DataFrame,
    cost_model: CostModel = BLUECHIP_KIS,
) -> BacktestResult:
    """동일가중 벤치마크 백테스트.

    회전율은 거의 0 (가격 변동 외엔 거래 없음) → 비용 영향 미미.
    """
    weights = equal_weight_universe(prices)
    return run_backtest(prices, weights, cost_model=cost_model)
