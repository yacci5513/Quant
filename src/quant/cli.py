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
app.add_typer(data_app)


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


if __name__ == "__main__":
    app()
