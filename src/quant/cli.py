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
) -> None:
    """KRX 일봉 데이터 수집 (KOSPI 200 또는 지정 종목)."""
    from quant.data.price.fetch_krx import fetch_all

    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else None
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


if __name__ == "__main__":
    app()
