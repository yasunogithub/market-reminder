#!/usr/bin/env python3
"""
Fetch Historical Data Script

This script fetches all available historical data for market indicators.
Run this once to bootstrap historical data, or periodically to update.

Usage:
    python fetch_historical.py

Environment Variables:
    FORCE_CFTC_HISTORICAL: Force re-fetch of CFTC historical data
"""
import io
import os
import sys
import zipfile

import pandas as pd
import requests

# Add parent dir for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "data"
CHART_DIR = "charts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


def fetch_csv(url: str, **kwargs) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), **kwargs)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("=" * 60)
    print("Historical Data Fetcher")
    print("=" * 60)

    results = {"success": [], "failed": []}

    # =========================================================================
    # 1. VIX (FRED) - Full historical from 1990
    # =========================================================================
    print("\n[1] VIX (FRED)")
    try:
        vix = fetch_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", na_values=["."])
        # Normalize column names
        vix.columns = [c.lower() for c in vix.columns]
        if 'date' not in vix.columns:
            vix = vix.rename(columns={vix.columns[0]: 'date'})
        vix_col = [c for c in vix.columns if 'vix' in c.lower()]
        if vix_col:
            vix = vix.rename(columns={vix_col[0]: 'vix'})
        else:
            non_date = [c for c in vix.columns if c != 'date']
            if non_date:
                vix = vix.rename(columns={non_date[0]: 'vix'})

        save_csv(vix[['date', 'vix']], os.path.join(DATA_DIR, "vix.csv"))
        vix_dates = pd.to_datetime(vix['date'], errors='coerce')
        print(f"  ✓ VIX: {len(vix)} rows, {vix_dates.min().date()} to {vix_dates.max().date()}")
        results["success"].append(f"VIX: {len(vix)} rows")
    except Exception as e:
        print(f"  ✗ VIX: {e}")
        results["failed"].append(f"VIX: {e}")

    # =========================================================================
    # 2. RSI for multiple targets
    # =========================================================================
    print("\n[2] RSI Targets")

    rsi_targets = [
        ("^spx", "S&P 500"),
        ("^ndq", "NASDAQ 100"),
        ("^dji", "Dow Jones"),
        ("^nkx", "Nikkei 225 (Stooq)"),
        ("fx.f", "Fear Index"),
        ("acwi.us", "MSCI ACWI"),
        ("spy.us", "SPY ETF"),
        ("qqq.us", "QQQ ETF"),
        ("eem.us", "EEM Emerging Markets"),
        ("iwm.us", "IWM Russell 2000"),
        ("gld.us", "GLD Gold"),
        ("tlt.us", "TLT Treasury 20+"),
    ]

    # Nikkei 225 Official
    print("  Fetching Nikkei 225 Official...")
    try:
        nikkei_url = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_en.csv"
        nikkei = fetch_csv(nikkei_url)
        # Find columns
        date_col = None
        close_col = None
        for c in nikkei.columns:
            if 'date' in str(c).lower():
                date_col = c
            if 'close' in str(c).lower() or 'closing' in str(c).lower():
                if close_col is None:
                    close_col = c
        if not close_col:
            for c in nikkei.columns:
                if 'nikkei' in str(c).lower():
                    close_col = c
                    break
        if date_col and close_col:
            px = nikkei[[date_col, close_col]].copy()
            px.columns = ['date', 'close']
            px['close'] = pd.to_numeric(px['close'].astype(str).str.replace(',', ''), errors='coerce')
            px['date'] = pd.to_datetime(px['date'], errors='coerce')
            px = px.dropna().sort_values('date')
            px['RSI14'] = rsi(px['close'], 14)
            save_csv(px, os.path.join(DATA_DIR, "NIKKEI_OFFICIAL_rsi.csv"))
            dates = px['date']
            print(f"  ✓ Nikkei Official: {len(px)} rows, {dates.min().date()} to {dates.max().date()}")
            results["success"].append(f"Nikkei Official: {len(px)} rows")
    except Exception as e:
        print(f"  ✗ Nikkei Official: {e}")
        results["failed"].append(f"Nikkei Official: {e}")

    # Stooq targets
    for symbol, label in rsi_targets:
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
            px = fetch_csv(url)

            # Normalize columns
            px.columns = [c.lower() for c in px.columns]
            if 'date' not in px.columns:
                print(f"  ✗ {label}: No 'date' column")
                continue
            if 'close' not in px.columns:
                print(f"  ✗ {label}: No 'close' column")
                continue

            px = px[['date', 'close']].copy()
            px['date'] = pd.to_datetime(px['date'], errors='coerce')
            px['close'] = pd.to_numeric(px['close'], errors='coerce')
            px = px.dropna().sort_values('date')
            px['RSI14'] = rsi(px['close'], 14)

            safe_sym = symbol.replace('^', '').replace('.', '_')
            save_csv(px, os.path.join(DATA_DIR, f"{safe_sym}_rsi.csv"))
            dates = px['date']
            print(f"  ✓ {label}: {len(px)} rows, {dates.min().date()} to {dates.max().date()}")
            results["success"].append(f"{label}: {len(px)} rows")
        except Exception as e:
            print(f"  ✗ {label}: {e}")
            results["failed"].append(f"{label}: {e}")

    # =========================================================================
    # 3. CFTC Historical (yearly archives 2006-present)
    # =========================================================================
    print("\n[3] CFTC Historical Data")
    print("  Fetching CFTC FinFut yearly archives (2006-present)...")

    cftc_hist_dfs = []
    from datetime import datetime
    current_year = datetime.now().year

    for year in range(2006, current_year + 1):
        try:
            url = f"https://www.cftc.gov/files/dea/history/fin_fut_txt_{year}.zip"
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for name in z.namelist():
                        if name.endswith('.txt'):
                            with z.open(name) as f:
                                df = pd.read_csv(f)
                                cftc_hist_dfs.append(df)
                                print(f"    ✓ {year}: {len(df)} rows")
            else:
                print(f"    - {year}: Not available (HTTP {r.status_code})")
        except Exception as e:
            print(f"    ✗ {year}: {e.__class__.__name__}")

    if cftc_hist_dfs:
        cftc_hist = pd.concat(cftc_hist_dfs, ignore_index=True)
        save_csv(cftc_hist, os.path.join(DATA_DIR, "cftc_finfut_historical.csv"))
        print(f"  ✓ CFTC Historical Total: {len(cftc_hist)} rows")
        results["success"].append(f"CFTC Historical: {len(cftc_hist)} rows")
    else:
        print("  ✗ No historical CFTC data retrieved")
        results["failed"].append("CFTC Historical: No data")

    # Current week
    print("  Fetching current week CFTC FinFutWk...")
    try:
        cftc_url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
        cftc = pd.read_csv(cftc_url, header=None)
        save_csv(cftc, os.path.join(DATA_DIR, "cftc_finfutwk_raw.csv"))
        print(f"  ✓ CFTC Current Week: {len(cftc)} contracts")
        results["success"].append(f"CFTC Current: {len(cftc)} contracts")
    except Exception as e:
        print(f"  ✗ CFTC current: {e}")
        results["failed"].append(f"CFTC Current: {e}")

    # =========================================================================
    # 4. AAII Sentiment
    # =========================================================================
    print("\n[4] AAII Sentiment")
    print("  Trying ibmetrics.com mirror...")
    try:
        tables = pd.read_html("https://ibmetrics.com/aaii-sentiment.html")
        found = False
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("bull" in c for c in cols) and any("bear" in c for c in cols):
                save_csv(t, os.path.join(DATA_DIR, "aaii.csv"))
                print(f"  ✓ AAII (mirror): {len(t)} rows")
                results["success"].append(f"AAII: {len(t)} rows")
                found = True
                break
        if not found:
            print("  ✗ AAII: No matching table found")
            results["failed"].append("AAII: No matching table")
    except Exception as e:
        print(f"  ✗ AAII mirror: {e}")
        results["failed"].append(f"AAII: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n✓ Success ({len(results['success'])}):")
    for s in results["success"]:
        print(f"  - {s}")

    if results["failed"]:
        print(f"\n✗ Failed ({len(results['failed'])}):")
        for f in results["failed"]:
            print(f"  - {f}")

    print("\n" + "=" * 60)
    print("Data Files in data/ directory:")
    print("=" * 60)

    import glob
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            fname = os.path.basename(f)
            print(f"  {fname}: {len(df)} rows")
        except Exception:
            pass

    print("\nDone!")


if __name__ == "__main__":
    main()
