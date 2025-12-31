import yfinance as yf
import pandas as pd


def fetch_data(ticker: str, period="5y", interval="1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False
    )

    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.sort_index()

    return df
