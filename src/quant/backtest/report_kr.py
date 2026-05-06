"""한국어 자체 HTML 리포트.

quantstats는 영문이라 텍스트 매핑 어려움. 자체 HTML 리포트 생성.
- 차트는 matplotlib (Latin 라벨로 폰트 의존성 회피)
- HTML/CSS는 한국어 (라벨, 해설, 가드레일 경고)
- 단일 파일 (PNG는 base64 인라인)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ReportSections:
    title: str
    period: tuple[str, str]
    metrics_strategy: dict
    metrics_benchmark: dict | None
    equity_strategy: pd.Series
    equity_benchmark: pd.Series | None
    daily_returns: pd.Series
    warnings: list[str]
    config_summary: dict | None = None


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _equity_chart(eq_strategy: pd.Series, eq_benchmark: pd.Series | None) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq_strategy.index, eq_strategy.values, label="strategy", linewidth=1.6, color="#2563eb")
    if eq_benchmark is not None:
        bench_norm = eq_benchmark / eq_benchmark.iloc[0]
        ax.plot(
            bench_norm.index,
            bench_norm.values,
            label="benchmark",
            linewidth=1.0,
            alpha=0.7,
            color="#94a3b8",
        )
    ax.set_title("Equity Curve (start = 1.0)")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return _fig_to_b64(fig)


def _drawdown_chart(eq: pd.Series) -> str:
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#ef4444", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#ef4444", linewidth=0.9)
    ax.set_title("Drawdown")
    ax.set_ylabel("DD")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_b64(fig)


def _monthly_heatmap(returns: pd.Series) -> str:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    monthly.index = pd.MultiIndex.from_arrays([monthly.index.year, monthly.index.month])
    pivot = monthly.unstack(level=-1)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [months[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, max(2.5, len(pivot) * 0.35)))
    data = pivot.values
    cmap = plt.get_cmap("RdYlGn")
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data))) if data.size else 0.01
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="monthly return")
    ax.set_title("Monthly Returns")
    return _fig_to_b64(fig)


_KOREAN_LABEL = {
    "cagr": ("연복리 수익률 (CAGR)", "%"),
    "total_return": ("누적 수익률", "%"),
    "sharpe": ("Sharpe (위험조정 수익)", ""),
    "sortino": ("Sortino (하락 위험조정)", ""),
    "max_drawdown": ("최대 낙폭 (MDD)", "%"),
    "calmar": ("Calmar (CAGR ÷ |MDD|)", ""),
    "volatility_annual": ("연환산 변동성", "%"),
    "n_trades": ("거래 횟수", "회"),
    "win_rate": ("승률", "%"),
    "turnover_annual": ("연간 회전율", "x"),
}


def _format_metric(key: str, value) -> str:
    label, unit = _KOREAN_LABEL.get(key, (key, ""))
    if value is None:
        return f"<tr><td>{label}</td><td>—</td></tr>"
    if unit == "%":
        return f"<tr><td>{label}</td><td>{value * 100:+.2f}%</td></tr>"
    if isinstance(value, int | np.integer):
        return f"<tr><td>{label}</td><td>{value:,d} {unit}</td></tr>"
    if isinstance(value, float):
        return f"<tr><td>{label}</td><td>{value:+.2f} {unit}</td></tr>"
    return f"<tr><td>{label}</td><td>{value}</td></tr>"


def _metrics_table(metrics: dict, title: str) -> str:
    rows = []
    for key in [
        "total_return",
        "cagr",
        "volatility_annual",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "n_trades",
        "win_rate",
        "turnover_annual",
    ]:
        if key in metrics:
            rows.append(_format_metric(key, metrics[key]))
    body = "\n".join(rows)
    return f"""<div class="metric-card">
  <h3>{title}</h3>
  <table class="metrics">{body}</table>
</div>"""


def render_html(sections: ReportSections, output_path: Path) -> Path:
    eq_b64 = _equity_chart(sections.equity_strategy, sections.equity_benchmark)
    dd_b64 = _drawdown_chart(sections.equity_strategy)
    heat_b64 = _monthly_heatmap(sections.daily_returns)

    metrics_strat = _metrics_table(sections.metrics_strategy, "전략 성과")
    metrics_bench = (
        _metrics_table(sections.metrics_benchmark, "벤치마크 성과")
        if sections.metrics_benchmark
        else ""
    )

    warning_block = ""
    if sections.warnings:
        items = "".join(f"<li>{w}</li>" for w in sections.warnings)
        warning_block = f"""<section class="warnings">
  <h2>가드레일 경고</h2>
  <ul>{items}</ul>
</section>"""

    config_block = ""
    if sections.config_summary:
        cfg_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sections.config_summary.items()
        )
        config_block = f"""<section class="config">
  <h2>전략 설정</h2>
  <table class="metrics">{cfg_rows}</table>
</section>"""

    p_start, p_end = sections.period
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sections.title}</title>
<style>
  body {{ font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
          max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #0f172a;
          background: #f8fafc; }}
  header {{ background: white; padding: 18px 22px; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 18px; }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
  .period {{ color: #64748b; font-size: 14px; }}
  section {{ background: white; padding: 16px 22px; border-radius: 12px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 18px; }}
  section h2 {{ margin: 4px 0 12px 0; font-size: 18px; color: #1e40af; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .metric-card h3 {{ margin: 4px 0 10px 0; font-size: 15px; color: #475569; }}
  table.metrics {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  table.metrics td {{ padding: 6px 8px; border-bottom: 1px solid #f1f5f9; }}
  table.metrics td:first-child {{ color: #64748b; }}
  table.metrics td:last-child {{ font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
  .warnings {{ border-left: 4px solid #f59e0b; }}
  .warnings ul {{ margin: 6px 0 0 18px; }}
  .warnings li {{ margin: 4px 0; color: #b45309; font-size: 14px; }}
  .footnote {{ color: #94a3b8; font-size: 12px; margin-top: 10px; text-align: right; }}
</style>
</head>
<body>

<header>
  <h1>{sections.title}</h1>
  <div class="period">분석 기간: {p_start} ~ {p_end}</div>
</header>

<section>
  <h2>1. 핵심 지표</h2>
  <div class="grid">
    {metrics_strat}
    {metrics_bench}
  </div>
</section>

<section>
  <h2>2. 자본 곡선 (Equity Curve)</h2>
  <img src="data:image/png;base64,{eq_b64}" alt="equity curve">
  <p style="font-size:13px;color:#475569;">
    초기 자본을 1.0으로 정규화한 자본 곡선입니다. 위로 갈수록 자본이 증가했음을 의미합니다.
  </p>
</section>

<section>
  <h2>3. 최대 낙폭 (Drawdown)</h2>
  <img src="data:image/png;base64,{dd_b64}" alt="drawdown">
  <p style="font-size:13px;color:#475569;">
    역대 최고점 대비 현재까지의 손실 비율입니다. 0%에 가까울수록 좋습니다.
    가드레일 §10: MDD ≤ -30% 이면 라이브 진입 금지.
  </p>
</section>

<section>
  <h2>4. 월별 수익률 히트맵</h2>
  <img src="data:image/png;base64,{heat_b64}" alt="monthly heatmap">
  <p style="font-size:13px;color:#475569;">
    초록 = 양의 수익, 빨강 = 음의 수익. 색상 강도가 클수록 큰 수익/손실.
  </p>
</section>

{warning_block}

{config_block}

<div class="footnote">
  Generated by Quant — 한국 주식 자동매매 시스템.
  본 리포트는 백테스트 결과로 미래 수익을 보장하지 않습니다.
  자동매매로 인한 손익은 투자자 본인의 책임입니다.
</div>

</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_sections_from_result(
    result,
    *,
    title: str,
    benchmark_result=None,
    config_summary: dict | None = None,
) -> ReportSections:
    """BacktestResult → ReportSections 어댑터."""
    m = result.metrics
    metrics_dict = {
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
    }
    bench_dict = None
    if benchmark_result is not None:
        bm = benchmark_result.metrics
        bench_dict = {
            "total_return": bm.total_return,
            "cagr": bm.cagr,
            "volatility_annual": bm.volatility_annual,
            "sharpe": bm.sharpe,
            "sortino": bm.sortino,
            "max_drawdown": bm.max_drawdown,
            "calmar": bm.calmar,
            "n_trades": bm.n_trades,
            "win_rate": bm.win_rate,
            "turnover_annual": bm.turnover_annual,
        }
    return ReportSections(
        title=title,
        period=(str(m.period_start), str(m.period_end)),
        metrics_strategy=metrics_dict,
        metrics_benchmark=bench_dict,
        equity_strategy=result.equity_curve,
        equity_benchmark=benchmark_result.equity_curve if benchmark_result else None,
        daily_returns=result.daily_returns,
        warnings=list(m.warnings),
        config_summary=config_summary,
    )
