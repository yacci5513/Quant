"""한국 주식 시장 거래비용 모델.

vectorbt에 주입할 fee/slippage 비율을 계산한다.
보수적 디폴트 — 백테스트는 실거래보다 비싸야 한다 (가드레일 §4).

비용 구조 (2025 기준):
- 거래세 (매도 시): KOSPI/KOSDAQ 0.18%
- KIS 비대면 매매수수료: 매수/매도 각 ~0.015% (보수적으로 설정 가능)
- 슬리피지: 우량주 0.05% / 중소형 0.1~0.5%

vectorbt의 from_signals는 fee를 '한쪽당' 비율로 받음.
거래세는 매도 시만 부과되지만, 단순화를 위해 양쪽 평균 비용으로 모델링하거나,
정밀하게 하려면 매도/매수 비율을 별도로 계산.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# 한국 시장 표준 비용 (2025)
KOSPI_TRANSFER_TAX: Final[float] = 0.0018  # 매도 시 0.18%
KOSDAQ_TRANSFER_TAX: Final[float] = 0.0018  # 매도 시 0.18%

# KIS 비대면 매매수수료 (보수적). 실제론 0.0036~0.015% 범위.
KIS_COMMISSION_PER_SIDE: Final[float] = 0.00015  # 한쪽 0.015%

# 슬리피지 보수적 디폴트
SLIPPAGE_BLUECHIP: Final[float] = 0.0005  # 0.05% — 우량주(KOSPI 200)
SLIPPAGE_SMALLCAP: Final[float] = 0.0020  # 0.20% — 중소형


@dataclass(frozen=True)
class CostModel:
    """거래비용 묶음.

    vectorbt.Portfolio.from_signals(fees=, slippage=) 에 주입.
    fees는 한쪽당 (매수, 매도 각각 적용). 거래세를 매도쪽에만 부과하는
    정밀 모델은 별도 함수로 후처리.
    """

    commission_per_side: float = KIS_COMMISSION_PER_SIDE
    slippage_per_side: float = SLIPPAGE_BLUECHIP
    transfer_tax: float = KOSPI_TRANSFER_TAX  # 매도쪽에만 적용

    @property
    def avg_fee_per_side(self) -> float:
        """단순화: 거래세를 양쪽 평균으로 분배해 fees에 통합.

        실제로는 매도 시만 거래세가 부과되지만, 왕복 회전 기준으로 보면
        매수 + 매도 합쳐서 한 번 부과되는 효과. 매수/매도가 비슷한 빈도면
        절반씩 쪼개도 회계적 결과는 유사.
        """
        return self.commission_per_side + (self.transfer_tax / 2)

    @property
    def round_trip_cost(self) -> float:
        """1회 왕복 (매수 → 매도) 총 비용 비율. 백테스트 검증용."""
        buy = self.commission_per_side + self.slippage_per_side
        sell = self.commission_per_side + self.slippage_per_side + self.transfer_tax
        return buy + sell

    def describe(self) -> str:
        return (
            f"CostModel("
            f"commission/side={self.commission_per_side * 100:.4f}%, "
            f"slippage/side={self.slippage_per_side * 100:.4f}%, "
            f"transfer_tax={self.transfer_tax * 100:.4f}%, "
            f"round_trip={self.round_trip_cost * 100:.3f}%)"
        )


# 자주 쓰는 프리셋
BLUECHIP_KIS: Final[CostModel] = CostModel(
    commission_per_side=KIS_COMMISSION_PER_SIDE,
    slippage_per_side=SLIPPAGE_BLUECHIP,
    transfer_tax=KOSPI_TRANSFER_TAX,
)
"""KOSPI 200 / 우량주, KIS 비대면 매매. 가장 일반적."""

SMALLCAP_KIS: Final[CostModel] = CostModel(
    commission_per_side=KIS_COMMISSION_PER_SIDE,
    slippage_per_side=SLIPPAGE_SMALLCAP,
    transfer_tax=KOSDAQ_TRANSFER_TAX,
)
"""중소형주, 슬리피지 4배. KOSDAQ 등."""

CONSERVATIVE: Final[CostModel] = CostModel(
    commission_per_side=0.0005,  # 0.05% (높게)
    slippage_per_side=0.0015,  # 0.15%
    transfer_tax=KOSPI_TRANSFER_TAX,
)
"""의심스러운 백테스트를 한 번 더 깎아내고 싶을 때 사용."""
