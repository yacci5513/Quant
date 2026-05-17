"""D 시나리오 (주간 1m Top-5) walk-forward 정직 검증.

목적:
- 직전 세션에서 D (주간 W-FRI / 1m lookback / Top-5) 단순 IS Sharpe 1.85 발견.
- 단일 5년 IS는 과적합·기간 운 가능 → walk-forward로 일관성 확인.
- 챔피언 (월간 12m Top-10 + MA100)과 동일 윈도우에서 직접 비교.

가드레일:
- engine.auto_shift_weights=True (기본) — t 시그널 → t+1 매수 (룩어헤드 차단)
- 거래비용 BLUECHIP_KIS 적용
- 윈도우별 OOS만 측정 (lookback 워밍업은 학습 구간으로 간주)
- 가드레일 §10 라이브 진입 차단 규칙 자동 표시
"""

from __future__ import annotations

import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import walk_forward_windows
from quant.backtest.regime import filter_weights
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights


def run_strategy_walkforward(
    label: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    train_years: int = 2,
    test_months: int = 6,
    step_months: int = 6,
) -> pd.DataFrame:
    start = prices.index[0].date()
    end = prices.index[-1].date()
    windows = walk_forward_windows(
        start=start,
        end=end,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
    )

    rows = []
    for i, w in enumerate(windows, 1):
        mask = (prices.index.date >= w.test_start) & (prices.index.date <= w.test_end)
        oos_dates = prices.index[mask]
        if len(oos_dates) < 5:
            continue
        res = run_backtest(
            prices.loc[oos_dates],
            weights.loc[oos_dates],
            cost_model=BLUECHIP_KIS,
        )
        m = res.metrics
        rows.append(
            {
                "strategy": label,
                "window": i,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "mdd": m.max_drawdown,
                "turnover": m.turnover_annual or 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "strategy": label,
        "n_windows": len(df),
        "sharpe_mean": df["sharpe"].mean(),
        "sharpe_std": df["sharpe"].std(ddof=0),
        "sharpe_min": df["sharpe"].min(),
        "sharpe_max": df["sharpe"].max(),
        "cagr_mean": df["cagr"].mean(),
        "cagr_std": df["cagr"].std(ddof=0),
        "mdd_mean": df["mdd"].mean(),
        "mdd_worst": df["mdd"].min(),
        "win_pct": (df["cagr"] > 0).mean(),
        "turnover_mean": df["turnover"].mean(),
    }


def main() -> None:
    print("=" * 80)
    print("D 시나리오 walk-forward 정직 검증 (lookahead fix 적용)")
    print("=" * 80)

    prices = load_close_panel()
    values = load_value_panel()
    print(f"가격 패널: {prices.shape}, 기간 {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # === D 시나리오: 주간 1m Top-5 ===
    cfg_d = MomentumTopNConfig(
        lookback_months=1,
        skip_months=0,
        top_n=5,
        rebalance_freq="W-FRI",
    )
    w_d = generate_weights(prices, values=values, config=cfg_d)

    # === D + MA100 시장 필터 (안전성 추가 측정) ===
    w_d_ma, _ = filter_weights(prices, w_d, ma_window=100, mode="binary")

    # === 챔피언 (현재 라이브 후보): 월간 12m Top-10 + MA100 ===
    cfg_champ = MomentumTopNConfig(
        lookback_months=12,
        skip_months=1,
        top_n=10,
        rebalance_freq="BMS",
    )
    w_champ = generate_weights(prices, values=values, config=cfg_champ)
    w_champ_ma, _ = filter_weights(prices, w_champ, ma_window=100, mode="binary")

    # === Walk-forward (2년 워밍업 / 6개월 OOS / 6개월 step) ===
    print("\n실행 중: D (주간 1m Top-5)...")
    df_d = run_strategy_walkforward("D (주간 1m Top-5)", prices, w_d)

    print("실행 중: D + MA100...")
    df_d_ma = run_strategy_walkforward("D + MA100", prices, w_d_ma)

    print("실행 중: 챔피언 (월간 12m Top-10 + MA100)...")
    df_champ = run_strategy_walkforward("챔피언 (월간 12m + MA100)", prices, w_champ_ma)

    all_df = pd.concat([df_d, df_d_ma, df_champ], ignore_index=True)

    # === 결과 표 ===
    print("\n" + "=" * 80)
    print("📊 윈도우별 OOS 결과")
    print("=" * 80)
    for label in all_df["strategy"].unique():
        sub = all_df[all_df["strategy"] == label]
        print(f"\n[{label}]")
        for _, r in sub.iterrows():
            print(
                f"  W{r.window} {r.test_start}~{r.test_end} | "
                f"CAGR {r.cagr * 100:+7.2f}% | Sharpe {r.sharpe:+5.2f} | "
                f"MDD {r.mdd * 100:6.2f}% | Turn {r.turnover:.1f}x"
            )

    # === Summary ===
    summaries = [
        summarize(df_d, "D (주간 1m Top-5)"),
        summarize(df_d_ma, "D + MA100"),
        summarize(df_champ, "챔피언 (월간 12m + MA100)"),
    ]
    print("\n" + "=" * 80)
    print("📈 Summary (walk-forward OOS 분포)")
    print("=" * 80)
    print(
        f"{'전략':<32} {'N':>3} {'Sharpe':>8} {'±std':>6} {'CAGR':>9} "
        f"{'MDDworst':>10} {'WIN%':>6} {'Turn':>6}"
    )
    print("-" * 88)
    for s in summaries:
        print(
            f"{s['strategy']:<32} {s['n_windows']:>3} "
            f"{s['sharpe_mean']:>+8.2f} {s['sharpe_std']:>6.2f} "
            f"{s['cagr_mean'] * 100:>+8.1f}% {s['mdd_worst'] * 100:>+9.1f}% "
            f"{s['win_pct'] * 100:>5.0f}% {s['turnover_mean']:>5.1f}x"
        )

    # === 가드레일 §10 라이브 진입 차단 규칙 평가 ===
    print("\n" + "=" * 80)
    print("🚨 가드레일 §10 라이브 진입 차단 평가")
    print("=" * 80)
    for s in summaries:
        flags = []
        if s["sharpe_mean"] < 0.5:
            flags.append("❌ OOS Sharpe < 0.5")
        if abs(s["mdd_worst"]) > 0.30:
            flags.append(f"❌ MDD worst {s['mdd_worst'] * 100:.1f}% > 30%")
        if s["win_pct"] < 0.5:
            flags.append(f"⚠️ Win pct {s['win_pct'] * 100:.0f}% < 50% (운 의존)")
        if s["sharpe_std"] > abs(s["sharpe_mean"]) * 1.5:
            flags.append("⚠️ Sharpe std/mean > 1.5 (윈도우 의존성 강함)")
        status = "✅ PASS" if not flags else "  /  ".join(flags)
        print(f"  {s['strategy']:<32} → {status}")


if __name__ == "__main__":
    main()
