"""매일 알림 통합 모듈 테스트 (네트워크 미사용)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.daily_signal import (
    AlertType,
    DailySignal,
    HoldingRow,
    compute_daily_signal,
    render_alert,
)
from quant.strategies.momentum_topn import MomentumTopNConfig


def _synth_panel(scenario: str = "uptrend") -> tuple[pd.DataFrame, pd.DataFrame]:
    """합성 시장 + 거래대금 패널.

    scenario:
      'uptrend' = 단조 증가 (시장 MA 위)
      'downtrend' = 단조 감소 (시장 MA 아래)
      'cross_down' = 처음 위, 후반 아래 (MA 깨짐)
    """
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    cols = [f"T{i:03d}" for i in range(20)]
    if scenario == "uptrend":
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.005, (len(idx), len(cols)))
        prices = 100 * (1 + pd.DataFrame(rets, index=idx, columns=cols)).cumprod()
    elif scenario == "downtrend":
        rng = np.random.default_rng(1)
        rets = rng.normal(-0.0008, 0.005, (len(idx), len(cols)))
        prices = 100 * (1 + pd.DataFrame(rets, index=idx, columns=cols)).cumprod()
    else:  # cross_down
        rng = np.random.default_rng(2)
        rets1 = rng.normal(0.001, 0.005, (300, len(cols)))
        rets2 = rng.normal(-0.003, 0.008, (len(idx) - 300, len(cols)))
        rets = np.concatenate([rets1, rets2])
        prices = 100 * (1 + pd.DataFrame(rets, index=idx, columns=cols)).cumprod()
    values = prices * 1e7
    return prices, values


def test_compute_normal_holding_uptrend() -> None:
    prices, values = _synth_panel("uptrend")
    cfg = MomentumTopNConfig(
        top_n=5, lookback_months=6, rebalance_freq="BMS", replace_threshold=0.0
    )
    ds = compute_daily_signal(prices=prices, values=values, config=cfg, ma_window=100)
    # 상승장 → NORMAL 또는 REBALANCE
    assert ds.alert_type in {AlertType.NORMAL, AlertType.REBALANCE}
    assert ds.market_state["in_uptrend"]


def test_compute_liquidate_downtrend() -> None:
    prices, values = _synth_panel("downtrend")
    cfg = MomentumTopNConfig(top_n=5, lookback_months=6, rebalance_freq="BMS")
    ds = compute_daily_signal(prices=prices, values=values, config=cfg, ma_window=100)
    # 하락장 → LIQUIDATE
    assert ds.alert_type is AlertType.LIQUIDATE
    assert not ds.market_state["in_uptrend"]


def test_render_alert_contains_market_state() -> None:
    signal = DailySignal(
        as_of=pd.Timestamp("2026-05-07").date(),
        alert_type=AlertType.NORMAL,
        market_state={
            "as_of": "2026-05-07",
            "market_value": 100000.0,
            "ma_value": 80000.0,
            "distance_pct": 25.0,
            "in_uptrend": True,
            "recommendation": "풀투자 (시장 > MA200)",
        },
        holdings=[
            HoldingRow(
                ticker="005930",
                name="삼성전자",
                weight=0.1,
                last_close=87800.0,
                target_value_won=1_000_000,
                target_shares=11,
            ),
        ],
        new_buys=[],
        sells=[],
        next_rebalance=pd.Timestamp("2026-06-01").date(),
        notes=[],
    )
    text = render_alert(signal)
    assert "2026-05-07" in text
    assert "삼성전자" in text
    assert "보유 유지" in text
    assert "+25.0%" in text or "25.0" in text


def test_render_alert_liquidate_shows_sells() -> None:
    signal = DailySignal(
        as_of=pd.Timestamp("2026-05-07").date(),
        alert_type=AlertType.LIQUIDATE,
        market_state={
            "as_of": "2026-05-07",
            "market_value": 100000.0,
            "ma_value": 110000.0,
            "distance_pct": -9.0,
            "in_uptrend": False,
            "recommendation": "현금화",
        },
        holdings=[],
        new_buys=[],
        sells=[
            HoldingRow(ticker="005930", name="삼성전자", weight=0.1, target_shares=11),
        ],
        next_rebalance=None,
        notes=["MA100 하향 돌파"],
    )
    text = render_alert(signal)
    assert "즉시 행동" in text
    assert "전량 청산" in text
    assert "삼성전자" in text


def test_render_alert_rebalance_shows_buys_and_sells() -> None:
    signal = DailySignal(
        as_of=pd.Timestamp("2026-05-01").date(),
        alert_type=AlertType.REBALANCE,
        market_state={
            "as_of": "2026-05-01",
            "market_value": 100000.0,
            "ma_value": 80000.0,
            "distance_pct": 25.0,
            "in_uptrend": True,
            "recommendation": "풀투자",
        },
        holdings=[],
        new_buys=[
            HoldingRow(
                ticker="000660",
                name="SK하이닉스",
                weight=0.1,
                target_value_won=1_000_000,
                target_shares=10,
            ),
        ],
        sells=[
            HoldingRow(ticker="005930", name="삼성전자", weight=0.1, target_shares=11),
        ],
        next_rebalance=pd.Timestamp("2026-06-01").date(),
        notes=[],
    )
    text = render_alert(signal)
    assert "리밸런싱" in text
    assert "삼성전자" in text  # 매도
    assert "SK하이닉스" in text  # 매수
