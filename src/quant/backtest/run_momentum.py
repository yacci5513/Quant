"""모멘텀 Top-N 백테스트 러너 (CLI에서 호출).

흐름:
  1. data/raw/prices/*.parquet 로드 (종가 + 거래대금)
  2. 모멘텀 시그널 → 가중치 패널 생성
  3. IS/OOS 분리 백테스트 (vectorbt 없이 자체 엔진 사용)
  4. 메트릭·가드레일 경고 출력
  5. (선택) CSV·PNG 저장

가드레일:
- 시점별 KOSPI 200 구성을 별도 추적하지 않으므로, 데이터에 존재하는 종목만 유니버스로 사용.
- 더 엄격하려면 매 리밸런싱 시점의 fetch_kospi200_tickers(d) 호출 (Phase 후속).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS, CostModel
from quant.backtest.engine import run_split
from quant.backtest.metrics import Metrics
from quant.common.config import get_settings
from quant.common.logger import logger
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

_COST_PRESETS: dict[str, CostModel] = {
    "bluechip": BLUECHIP_KIS,
    "smallcap": SMALLCAP_KIS,
    "conservative": CONSERVATIVE,
}


def run(
    *,
    top_n: int = 10,
    lookback_months: int = 12,
    skip_months: int = 1,
    min_value: float = 1e9,
    is_ratio: float = 0.7,
    cost_preset: str = "bluechip",
    save: bool = True,
) -> None:
    settings = get_settings()
    out_dir = settings.data_dir / "backtest" / "momentum_topn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    logger.info("데이터 로딩...")
    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        logger.error("저장된 가격 데이터 없음 — `quant data fetch-krx`로 먼저 수집")
        raise SystemExit(1)
    logger.info(
        f"가격 패널: {prices.shape[0]} days × {prices.shape[1]} tickers "
        f"({prices.index[0].date()} ~ {prices.index[-1].date()})"
    )

    # 2. 가중치 생성
    cfg = MomentumTopNConfig(
        lookback_months=lookback_months,
        skip_months=skip_months,
        top_n=top_n,
        min_avg_value=min_value,
    )
    weights = generate_weights(prices, values=values, config=cfg)

    # 3. 비용 모델
    cost = _COST_PRESETS.get(cost_preset, BLUECHIP_KIS)
    logger.info(cost.describe())

    # 4. IS/OOS 분리 백테스트
    is_res, oos_res = run_split(prices, weights, is_ratio=is_ratio, cost_model=cost)

    # 5. 리포트 출력
    _print_section("In-Sample", is_res.metrics, is_res.n_rebalances, is_res.avg_n_holdings)
    _print_section("Out-of-Sample", oos_res.metrics, oos_res.n_rebalances, oos_res.avg_n_holdings)
    _print_validation(is_res.metrics, oos_res.metrics)

    # 6. 저장
    if save:
        _save_results(out_dir, prices, weights, is_res, oos_res, cfg, cost)
        _save_plots(out_dir, is_res, oos_res)
        logger.info(f"결과 저장: {out_dir}")


def _print_section(label: str, m: Metrics, n_rebal: int, avg_holdings: float) -> None:
    bar = "=" * 60
    logger.info(f"\n{bar}\n{label}\n{bar}")
    logger.info(f"\n{m.summary()}리밸런싱 횟수: {n_rebal}, 평균 보유: {avg_holdings:.1f}종목")
    if m.warnings:
        for w in m.warnings:
            logger.warning(w)


def _print_validation(is_m: Metrics, oos_m: Metrics) -> None:
    """가드레일 §1: IS vs OOS 비교 — Sharpe 갭이 너무 크면 과적합 의심."""
    bar = "=" * 60
    logger.info(f"\n{bar}\nIS / OOS 검증\n{bar}")
    sharpe_gap = is_m.sharpe - oos_m.sharpe
    cagr_gap = is_m.cagr - oos_m.cagr
    logger.info(f"Sharpe gap (IS-OOS): {sharpe_gap:+.2f}   |   CAGR gap: {cagr_gap * 100:+.2f}%")
    if oos_m.sharpe < 0.5:
        logger.warning(f"🚨 OOS Sharpe {oos_m.sharpe:.2f} < 0.5 — 라이브 진입 금지 (가드레일 §10)")
    if sharpe_gap > 1.0 and is_m.sharpe > 0:
        logger.warning("⚠️ Sharpe gap > 1.0 — IS에선 잘 작동했지만 OOS에서 무너짐. 과적합 가능성.")
    if oos_m.max_drawdown < -0.30:
        logger.warning(f"🚨 OOS MDD {oos_m.max_drawdown * 100:.1f}% < -30% — 라이브 진입 금지.")


def _save_results(
    out_dir: Path,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    is_res,
    oos_res,
    cfg: MomentumTopNConfig,
    cost: CostModel,
) -> None:
    is_res.equity_curve.rename("equity").to_csv(out_dir / "equity_is.csv")
    oos_res.equity_curve.rename("equity").to_csv(out_dir / "equity_oos.csv")
    is_res.daily_returns.rename("ret").to_csv(out_dir / "returns_is.csv")
    oos_res.daily_returns.rename("ret").to_csv(out_dir / "returns_oos.csv")
    weights.to_parquet(out_dir / "weights.parquet")

    summary = {
        "config": {
            "top_n": cfg.top_n,
            "lookback_months": cfg.lookback_months,
            "skip_months": cfg.skip_months,
            "min_avg_value": cfg.min_avg_value,
            "rebalance_freq": cfg.rebalance_freq,
        },
        "cost": {
            "commission_per_side": cost.commission_per_side,
            "slippage_per_side": cost.slippage_per_side,
            "transfer_tax": cost.transfer_tax,
            "round_trip_cost": cost.round_trip_cost,
        },
        "is": _metrics_to_dict(is_res.metrics, is_res.n_rebalances, is_res.avg_n_holdings),
        "oos": _metrics_to_dict(oos_res.metrics, oos_res.n_rebalances, oos_res.avg_n_holdings),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def _save_plots(out_dir: Path, is_res, oos_res) -> None:
    from quant.backtest.plot import save_all

    paths = save_all(
        out_dir / "plots",
        equity_is=is_res.equity_curve,
        equity_oos=oos_res.equity_curve,
        returns_is=is_res.daily_returns,
        returns_oos=oos_res.daily_returns,
    )
    for name, p in paths.items():
        logger.info(f"  plot {name}: {p}")


def _metrics_to_dict(m: Metrics, n_rebal: int, avg_holdings: float) -> dict:
    return {
        "period": [str(m.period_start), str(m.period_end)],
        "days": m.days,
        "total_return": m.total_return,
        "cagr": m.cagr,
        "volatility_annual": m.volatility_annual,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar,
        "n_trades": m.n_trades,
        "win_rate": m.win_rate,
        "turnover_annual": m.turnover_annual,
        "n_rebalances": n_rebal,
        "avg_holdings": avg_holdings,
        "warnings": m.warnings,
    }
