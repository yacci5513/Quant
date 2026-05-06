"""다중 시그널 결합 (앙상블).

여러 전략의 가중치 패널을 정의된 비중으로 가중 평균.
각 전략은 자기 빈도로 작동 → 합치면 매주 뭔가 일어나는 효과.

가드레일 §1: 결합 비중도 파라미터 → 과적합 위험. 의미 있는 다양화일 때만 사용.
가드레일 §5: Crowding 분산 효과 — 상관관계 낮은 전략끼리 결합해야 의미.

설계:
- 각 전략 가중치 패널은 합 1.0 또는 0.0 (전략별 풀 노출)
- 결합: weights_combined = Σ (alloc_i × weights_i)
- 결과 합도 1.0 (또는 일부 전략이 비어있는 시점은 < 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class StrategyAllocation:
    """단일 전략의 결합 비중."""

    name: str
    weights: pd.DataFrame
    allocation: float  # 0~1 사이. 전체 알로케이션 합 = 1.0


def combine(
    allocations: list[StrategyAllocation],
    *,
    normalize_each: bool = True,
) -> pd.DataFrame:
    """여러 전략 가중치를 비중으로 결합.

    Args:
        allocations: 전략별 (이름, 가중치 패널, 비중) 리스트
        normalize_each: True면 각 전략 가중치를 매일 합 1.0으로 재정규화 (전략 간 균등 노출 보장)

    Returns:
        결합 가중치 패널 (index/columns 합집합)
    """
    if not allocations:
        raise ValueError("결합할 전략이 없음")

    total_alloc = sum(a.allocation for a in allocations)
    if abs(total_alloc - 1.0) > 1e-6:
        logger.warning(f"앙상블 비중 합 {total_alloc:.4f} ≠ 1.0 — 정규화함")
        allocations = [
            StrategyAllocation(a.name, a.weights, a.allocation / total_alloc) for a in allocations
        ]

    # 인덱스/컬럼 합집합으로 정렬
    all_idx = sorted(set().union(*[a.weights.index for a in allocations]))
    all_cols = sorted(set().union(*[a.weights.columns for a in allocations]))

    combined = pd.DataFrame(0.0, index=all_idx, columns=all_cols, dtype=float)
    for a in allocations:
        w = a.weights.reindex(index=all_idx, columns=all_cols).fillna(0.0)
        if normalize_each:
            row_sum = w.sum(axis=1)
            # 합이 0인 날은 그대로 0
            w = w.div(row_sum.where(row_sum > 0, 1), axis=0)
        combined = combined + a.allocation * w

    logger.info(
        "앙상블 결합 완료: " + ", ".join(f"{a.name}={a.allocation * 100:.0f}%" for a in allocations)
    )
    return combined
