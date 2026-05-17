"""백테스트용 손절·익절·트레일링 시뮬레이션.

가중치 패널에 종목별 손절/익절/트레일링을 적용한 새 가중치 패널을 반환.
이걸 기존 `run_backtest`에 그대로 넘기면 손절·익절 반영된 결과 얻음.

가정:
- 트리거 hit 시 같은 날 종가에 매도 (해당 날부터 가중치 0)
- 다음 진입 (가중치 0 → 양수)에 entry_price/max_pp 리셋
- spare 종목 자동 채움은 미적용 (현금화)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StopConfig:
    stop_loss_pct: float = -10.0  # %
    take_profit_pct: float = 20.0  # %
    trail_drawdown_pct: float = 10.0  # 최고 대비 -% 후퇴
    trail_activation_pct: float = 10.0  # 최고 손익률 활성화 임계


def apply_stops(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: StopConfig | None = None,
) -> pd.DataFrame:
    """가중치 패널 → 손절·익절·트레일링 적용한 새 가중치 패널.

    종목별로 진입~청산 구간을 추적하면서 매일 손익률 계산 후 트리거 시
    그 시점부터 다음 진입까지 가중치 0으로 만든다.
    """
    cfg = config or StopConfig()
    w = weights.copy().fillna(0.0)
    px = prices.reindex(index=w.index, columns=w.columns)

    for ticker in w.columns:
        wcol = w[ticker].to_numpy(copy=True)
        pcol = px[ticker].to_numpy()
        n = len(wcol)
        if not (wcol > 0).any():
            continue

        entry_price: float | None = None
        max_pp: float | None = None
        forced_exit = False

        for i in range(n):
            w_today = wcol[i]
            p_today = pcol[i]
            w_prev = wcol[i - 1] if i > 0 else 0.0

            # 새 진입 (이전 0, 오늘 양수)
            if w_today > 0 and w_prev <= 0:
                entry_price = p_today if p_today > 0 else None
                max_pp = 0.0
                forced_exit = False
                continue

            # 강제 청산 상태 유지
            if forced_exit:
                if w_today > 0:
                    wcol[i] = 0.0
                continue

            # 보유 중 — 트리거 체크
            if w_today > 0 and entry_price is not None and entry_price > 0:
                pp = (p_today - entry_price) / entry_price * 100.0
                if max_pp is None or pp > max_pp:
                    max_pp = pp
                hit_stop = pp <= cfg.stop_loss_pct
                hit_take = pp >= cfg.take_profit_pct
                hit_trail = (
                    max_pp >= cfg.trail_activation_pct and pp <= max_pp - cfg.trail_drawdown_pct
                )
                if hit_stop or hit_take or hit_trail:
                    wcol[i] = 0.0
                    forced_exit = True
                    entry_price = None
                    max_pp = None
            elif w_today == 0:
                entry_price = None
                max_pp = None

        w[ticker] = wcol
    return w
