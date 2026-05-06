"""Quant CLI 진입점.

`quant <subcommand>` 형태. 서브커맨드 그룹은 typer로 묶는다.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="quant",
    help="한국 주식 자동매매 시스템 CLI",
    no_args_is_help=True,
)

data_app = typer.Typer(name="data", help="데이터 수집/관리")
backtest_app = typer.Typer(name="backtest", help="백테스트 실행/분석")
app.add_typer(data_app)
app.add_typer(backtest_app)


# ----- 루트 -----
@app.command()
def version() -> None:
    """패키지 버전 출력."""
    from importlib.metadata import version as _v

    typer.echo(_v("quant"))


@app.command()
def hello() -> None:
    """동작 확인용 핑."""
    typer.echo("quant CLI: ok")


# ----- data -----
@data_app.command("fetch-krx")
def data_fetch_krx(
    years: int = typer.Option(5, "--years", "-y", help="가져올 기간(년)"),
    tickers: str | None = typer.Option(
        None,
        "--tickers",
        "-t",
        help="콤마구분 종목코드 (생략 시 KOSPI 200 전체)",
    ),
    kosdaq_top: int = typer.Option(
        0,
        "--kosdaq-top",
        help="KOSDAQ 시총 상위 N 추가 (0=KOSPI만, 100=꿀종목 후보군 확장)",
    ),
) -> None:
    """KRX 일봉 데이터 수집 (KOSPI 200 + 선택적 KOSDAQ Top-N)."""
    from quant.data.price.fetch_krx import fetch_all, fetch_universe_tickers

    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
    elif kosdaq_top > 0:
        ticker_list = fetch_universe_tickers(kosdaq_top=kosdaq_top)
        typer.echo(f"통합 유니버스: KOSPI 200 + KOSDAQ {kosdaq_top} = {len(ticker_list)}종목")
    else:
        ticker_list = None  # fetch_all 디폴트 = KOSPI 200
    fetch_all(tickers=ticker_list, years=years)


# ----- backtest -----
@backtest_app.command("momentum")
def backtest_momentum(
    top_n: int = typer.Option(10, "--top-n", "-n", help="상위 N 종목"),
    lookback: int = typer.Option(12, "--lookback", "-l", help="모멘텀 윈도우(개월)"),
    skip: int = typer.Option(1, "--skip", "-s", help="최근 제외 개월"),
    min_value_won: float = typer.Option(
        1e9, "--min-value", help="유동성 필터: 일평균 거래대금 임계(원)"
    ),
    is_ratio: float = typer.Option(0.7, "--is-ratio", help="In-Sample 비율"),
    cost_preset: str = typer.Option(
        "bluechip", "--cost", help="비용 프리셋: bluechip|smallcap|conservative"
    ),
    save_report: bool = typer.Option(True, "--save/--no-save", help="결과 CSV/PNG 저장 여부"),
) -> None:
    """KOSPI 200 모멘텀 Top-N 월 리밸런싱 백테스트."""
    from quant.backtest.run_momentum import run

    run(
        top_n=top_n,
        lookback_months=lookback,
        skip_months=skip,
        min_value=min_value_won,
        is_ratio=is_ratio,
        cost_preset=cost_preset,
        save=save_report,
    )


@backtest_app.command("walkforward")
def backtest_walkforward(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    skip: int = typer.Option(1, "--skip", "-s"),
    train_years: int = typer.Option(1, "--train-years", help="학습 윈도우(년)"),
    test_months: int = typer.Option(6, "--test-months", help="OOS 테스트 윈도우(개월)"),
    step_months: int = typer.Option(6, "--step-months", help="윈도우 슬라이딩 간격(개월)"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """Walk-forward 검증 (rolling 윈도우별 OOS Sharpe·CAGR 분포)."""
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.backtest.walkforward import run_walk_forward
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음 — quant data fetch-krx 먼저 실행")
        raise typer.Exit(1)

    weights = generate_weights(
        prices,
        values=values,
        config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback, skip_months=skip),
    )
    summary = run_walk_forward(
        prices,
        weights,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
        cost_model=cost,
    )

    out_dir = get_settings().data_dir / "backtest" / "walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.per_window.to_csv(out_dir / "windows.csv", index=False)
    typer.echo(f"\nresults saved to {out_dir / 'windows.csv'}")


@backtest_app.command("sweep")
def backtest_sweep(
    top_n_list: str = typer.Option("5,10,20", "--top-n-grid"),
    lookback_list: str = typer.Option("3,6,12", "--lookback-grid"),
    skip: int = typer.Option(1, "--skip", "-s"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """파라미터 그리드 스윕 (top_n × lookback)."""
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.backtest.plot import plot_sweep_heatmap
    from quant.backtest.sweep import run_sweep
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음 — quant data fetch-krx 먼저 실행")
        raise typer.Exit(1)

    top_n_grid = [int(x) for x in top_n_list.split(",")]
    lookback_grid = [int(x) for x in lookback_list.split(",")]

    res = run_sweep(
        prices,
        values=values,
        top_n_grid=top_n_grid,
        lookback_grid=lookback_grid,
        skip_months=skip,
        cost_model=cost,
    )

    out_dir = get_settings().data_dir / "backtest" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.df.to_csv(out_dir / "sweep_results.csv", index=False)

    fig = plot_sweep_heatmap(res.pivot("sharpe"), title="Sharpe by (top_n × lookback)")
    fig.savefig(out_dir / "sweep_sharpe.png", dpi=120)
    fig2 = plot_sweep_heatmap(res.pivot("cagr"), title="CAGR by (top_n × lookback)", cmap="RdYlGn")
    fig2.savefig(out_dir / "sweep_cagr.png", dpi=120)
    typer.echo(f"\nresults saved to {out_dir}/")


@backtest_app.command("compare")
def backtest_compare(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    skip: int = typer.Option(1, "--skip", "-s"),
    mr_short_days: int = typer.Option(5, "--mr-short-days"),
    mr_long_months: int = typer.Option(12, "--mr-long-months"),
    mr_top_n: int = typer.Option(10, "--mr-top-n"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """모멘텀 vs 평균회귀 vs 결합 vs 벤치마크 비교."""
    from quant.backtest.compare import run_compare
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.mean_reversion import MeanReversionConfig
    from quant.strategies.momentum_topn import MomentumTopNConfig

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음")
        raise typer.Exit(1)

    out_dir = get_settings().data_dir / "backtest" / "compare"
    run_compare(
        prices,
        values=values,
        momentum_cfg=MomentumTopNConfig(top_n=top_n, lookback_months=lookback, skip_months=skip),
        meanrev_cfg=MeanReversionConfig(
            short_lookback_days=mr_short_days,
            long_lookback_months=mr_long_months,
            top_n=mr_top_n,
        ),
        cost_model=cost,
        out_dir=out_dir,
    )


@app.command()
def signal(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    skip: int = typer.Option(1, "--skip", "-s"),
    seed: float | None = typer.Option(
        None,
        "--seed",
        "-S",
        help="시드 금액(원). 입력 시 종목별 매수 금액·수량 계산.",
    ),
) -> None:
    """현재 시점 모멘텀 시그널 — 오늘 매수해야 할 종목 N개 출력."""
    from quant.signal import latest_momentum_signal, render_signal_table
    from quant.strategies.momentum_topn import MomentumTopNConfig

    rows = latest_momentum_signal(
        config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback, skip_months=skip),
        seed_won=seed,
    )
    typer.echo("\n" + render_signal_table(rows, seed_won=seed))


@backtest_app.command("report")
def backtest_report(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    skip: int = typer.Option(1, "--skip", "-s"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """quantstats HTML 풀 리포트 생성 (전략 + 벤치마크 비교)."""
    from quant.backtest.benchmark import equal_weight_universe
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.backtest.engine import run_backtest
    from quant.backtest.report import generate_html
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음")
        raise typer.Exit(1)

    weights = generate_weights(
        prices,
        values=values,
        config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback, skip_months=skip),
    )
    res_strategy = run_backtest(prices, weights, cost_model=cost)
    res_bench = run_backtest(prices, equal_weight_universe(prices), cost_model=cost)

    out_dir = get_settings().data_dir / "backtest" / "report"
    out_path = out_dir / "momentum_report.html"
    generate_html(
        res_strategy.daily_returns,
        out_path,
        title=f"Momentum Top-{top_n} (lookback {lookback}m) vs Equal-weight Benchmark",
        benchmark=res_bench.daily_returns,
    )
    typer.echo(f"\nHTML 리포트 저장: {out_path}")


@backtest_app.command("history")
def backtest_history(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    skip: int = typer.Option(1, "--skip", "-s"),
) -> None:
    """리밸런싱별 보유 종목 + 다음 달 수익률 + 종목별 누적 통계."""
    from quant.backtest.audit import save_log
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음")
        raise typer.Exit(1)
    weights = generate_weights(
        prices,
        values=values,
        config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback, skip_months=skip),
    )
    out_dir = get_settings().data_dir / "backtest" / "history"
    paths = save_log(out_dir, prices, weights)
    typer.echo(f"\nrebalance log: {paths['log']}\nticker summary: {paths['ticker_summary']}")


@backtest_app.command("ensemble")
def backtest_ensemble(
    momentum_alloc: float = typer.Option(0.6, "--mom-alloc", help="모멘텀 비중"),
    lowvol_alloc: float = typer.Option(0.4, "--lv-alloc", help="저변동성 비중"),
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """모멘텀 + 저변동성 앙상블 백테스트 (분산 효과 확인용)."""
    from quant.backtest.benchmark import equal_weight_universe
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.backtest.engine import run_backtest
    from quant.backtest.ensemble import StrategyAllocation, combine
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.low_volatility import LowVolatilityConfig
    from quant.strategies.low_volatility import generate_weights as lv_w
    from quant.strategies.momentum_topn import MomentumTopNConfig
    from quant.strategies.momentum_topn import generate_weights as mom_w

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음 — quant data fetch-krx 먼저 실행")
        raise typer.Exit(1)

    mom = mom_w(
        prices, values=values, config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback)
    )
    lv = lv_w(prices, values=values, config=LowVolatilityConfig(top_n=top_n))
    combined = combine(
        [
            StrategyAllocation("momentum", mom, momentum_alloc),
            StrategyAllocation("low_vol", lv, lowvol_alloc),
        ]
    )
    res = run_backtest(prices, combined, cost_model=cost)
    bench = run_backtest(prices, equal_weight_universe(prices), cost_model=cost)
    typer.echo(f"\n앙상블 ({momentum_alloc * 100:.0f}/{lowvol_alloc * 100:.0f}):")
    typer.echo(res.metrics.summary())
    typer.echo("\n벤치마크:")
    typer.echo(bench.metrics.summary())


@backtest_app.command("report-kr")
def backtest_report_kr(
    top_n: int = typer.Option(10, "--top-n", "-n"),
    lookback: int = typer.Option(12, "--lookback", "-l"),
    cost_preset: str = typer.Option("bluechip", "--cost"),
) -> None:
    """한국어 HTML 리포트 (자체 generator, 가드레일 경고 포함)."""
    from quant.backtest.benchmark import equal_weight_universe
    from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE, SMALLCAP_KIS
    from quant.backtest.engine import run_backtest
    from quant.backtest.report_kr import build_sections_from_result, render_html
    from quant.common.config import get_settings
    from quant.data.price.fetch_krx import load_close_panel, load_value_panel
    from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

    presets = {"bluechip": BLUECHIP_KIS, "smallcap": SMALLCAP_KIS, "conservative": CONSERVATIVE}
    cost = presets.get(cost_preset, BLUECHIP_KIS)

    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        typer.echo("저장된 데이터 없음 — quant data fetch-krx 먼저 실행")
        raise typer.Exit(1)

    weights = generate_weights(
        prices, values=values, config=MomentumTopNConfig(top_n=top_n, lookback_months=lookback)
    )
    res = run_backtest(prices, weights, cost_model=cost)
    bench = run_backtest(prices, equal_weight_universe(prices), cost_model=cost)

    sections = build_sections_from_result(
        res,
        title=f"모멘텀 Top-{top_n} ({lookback}개월 lookback) 한국어 리포트",
        benchmark_result=bench,
        config_summary={
            "전략": "모멘텀 Top-N",
            "top_n": top_n,
            "lookback": f"{lookback}개월",
            "리밸런싱": "월 1회 (BMS)",
            "비용 프리셋": cost_preset,
        },
    )
    out_dir = get_settings().data_dir / "backtest" / "report_kr"
    out = render_html(sections, out_dir / "momentum_kr.html")
    typer.echo(f"\n한국어 리포트: {out}")


if __name__ == "__main__":
    app()
