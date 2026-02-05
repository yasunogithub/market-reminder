import io
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
CHART_DIR = "charts"
INPUT_DIR = "inputs"

NIKKEI_OFFICIAL_SYMBOL = "NIKKEI_OFFICIAL"


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_line_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    out_png: str,
) -> bool:
    d = df[[date_col, value_col]].dropna().copy()
    if d.empty:
        return False
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    if d.empty:
        return False
    plt.figure(figsize=(8, 4))
    plt.plot(d[date_col], d[value_col])
    plt.title(title)
    plt.xlabel(date_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def save_combined_chart(
    combined: pd.DataFrame,
    out_png: str,
    days: int = 365,
) -> bool:
    """Create a single-panel chart with all indicators overlaid."""
    if combined.empty or "date" not in combined.columns:
        return False

    df = combined.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Filter to recent N days
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    df = df[df["date"] >= cutoff]

    if df.empty:
        return False

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Left Y-axis: VIX, AAII, RSI, Margin (0-80 scale)
    # VIX - blue line
    if "vix" in df.columns and df["vix"].notna().any():
        d = df[["date", "vix"]].dropna()
        ax1.plot(d["date"], d["vix"], label="VIX", color="blue", linewidth=1.5)

    # AAII Bullish - red scatter
    if "AAII_Bullish" in df.columns and df["AAII_Bullish"].notna().any():
        d = df[["date", "AAII_Bullish"]].dropna()
        ax1.scatter(d["date"], d["AAII_Bullish"], label="AAII_Bullish", color="red", s=15, alpha=0.7)

    # Margin - yellow/orange scatter
    if "Margin" in df.columns and df["Margin"].notna().any():
        d = df[["date", "Margin"]].dropna()
        ax1.scatter(d["date"], d["Margin"], label="Margin", color="orange", s=15, alpha=0.7, marker="s")

    # RSI - green line
    if "RSI" in df.columns and df["RSI"].notna().any():
        d = df[["date", "RSI"]].dropna()
        ax1.plot(d["date"], d["RSI"], label="RSI", color="green", linewidth=1.5)

    ax1.set_ylabel("VIX / AAII / RSI / Margin")
    ax1.set_ylim(0, 80)
    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.3)

    # Right Y-axis: CFTC Net (inverted, 0 to -200000)
    ax2 = ax1.twinx()
    if "CFTC_Net" in df.columns and df["CFTC_Net"].notna().any():
        d = df[["date", "CFTC_Net"]].dropna()
        ax2.scatter(d["date"], d["CFTC_Net"], label="CFTC_Net", color="darkorange", s=15, alpha=0.7)
        ax2.set_ylabel("CFTC_Net", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        # Set y-limits to show negative values prominently
        min_val = d["CFTC_Net"].min()
        max_val = d["CFTC_Net"].max()
        ax2.set_ylim(min(min_val * 1.1, -200000), max(max_val * 1.1, 0))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.12))

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y/%m/%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def build_combined_df(
    vix_df: pd.DataFrame | None,
    aaii_df: pd.DataFrame | None,
    cftc_df: pd.DataFrame | None,
    margin_df: pd.DataFrame | None,
    rsi_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge all indicator DataFrames by date."""
    # Extract CFTC S&P 500 net position first
    cftc_sp500 = extract_cftc_sp500_net(cftc_df) if cftc_df is not None else None

    # Start with a date range
    all_dates: set[str] = set()

    def extract_dates(df: pd.DataFrame | None, date_col: str = "date") -> None:
        if df is None or df.empty:
            return
        col = None
        for c in df.columns:
            if "date" in str(c).lower():
                col = c
                break
        if col is None:
            return
        dates = pd.to_datetime(df[col], errors="coerce").dropna()
        all_dates.update(d.strftime("%Y-%m-%d") for d in dates)

    extract_dates(vix_df)
    extract_dates(aaii_df)
    extract_dates(cftc_sp500)
    extract_dates(margin_df)
    extract_dates(rsi_df)

    if not all_dates:
        return pd.DataFrame()

    combined = pd.DataFrame({"date": sorted(all_dates)})
    combined["date"] = pd.to_datetime(combined["date"])

    def merge_col(df: pd.DataFrame | None, src_col: str, dst_col: str) -> None:
        nonlocal combined
        if df is None or df.empty:
            return
        date_col = None
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break
        if date_col is None or src_col not in df.columns:
            return
        tmp = df[[date_col, src_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp = tmp.rename(columns={date_col: "date", src_col: dst_col})
        combined = combined.merge(tmp, on="date", how="left")

    # VIX
    merge_col(vix_df, "vix", "vix")

    # AAII
    if aaii_df is not None:
        for src, dst in [("bullish", "AAII_Bullish"), ("neutral", "AAII_Neutral"), ("bearish", "AAII_Bearish")]:
            for c in aaii_df.columns:
                if src in str(c).lower():
                    merge_col(aaii_df, c, dst)
                    break

    # CFTC - use pre-extracted S&P 500 net position
    merge_col(cftc_sp500, "CFTC_Net", "CFTC_Net")

    # Margin
    if margin_df is not None:
        for c in margin_df.columns:
            if "margin" in str(c).lower() or "balance" in str(c).lower():
                merge_col(margin_df, c, "Margin")
                break

    # RSI
    if rsi_df is not None:
        merge_col(rsi_df, "RSI14", "RSI")

    return combined.sort_values("date")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def slack_notify(
    webhook_url: str,
    text: str,
    image_urls: list[str] | None = None,
) -> None:
    if not webhook_url:
        return

    # Build payload with Block Kit if images provided
    if image_urls:
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        ]
        for url in image_urls:
            blocks.append({
                "type": "image",
                "image_url": url,
                "alt_text": "Chart",
            })
        payload = {"blocks": blocks, "text": text}  # text is fallback
    else:
        payload = {"text": text}

    r = requests.post(webhook_url, json=payload, timeout=20)
    r.raise_for_status()


def changed_since_last_run(tag: str, new_key: str) -> bool:
    """Save last seen key in data/_state.json and return update status."""
    state_path = os.path.join(DATA_DIR, "_state.json")
    state: dict[str, str] = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except json.JSONDecodeError:
            state = {}
    old = state.get(tag)
    state[tag] = new_key
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return old != new_key


# -----------------------
# Fetchers
# -----------------------

def fetch_csv(url: str, **read_csv_kwargs) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), **read_csv_kwargs)


def fetch_text_as_csv(url: str, header=None) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), header=header)


def fetch_stooq(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    return fetch_csv(url)


def fetch_vix_fred() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    df = fetch_csv(url, na_values=["."])

    # Detect date column (case-insensitive)
    date_col = _detect_date_col(list(df.columns))
    if not date_col:
        raise RuntimeError(f"VIX FRED CSV date column not found: {list(df.columns)}")

    # Detect VIX value column (VIXCLS or similar)
    vix_col = None
    for c in df.columns:
        if "vix" in str(c).lower():
            vix_col = c
            break
    if not vix_col:
        # Fallback: use the non-date column
        non_date_cols = [c for c in df.columns if c != date_col]
        if non_date_cols:
            vix_col = non_date_cols[0]
    if not vix_col:
        raise RuntimeError(f"VIX FRED CSV value column not found: {list(df.columns)}")

    return df.rename(columns={date_col: "date", vix_col: "vix"})


def fetch_cftc_finfutwk() -> pd.DataFrame:
    url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    return fetch_text_as_csv(url, header=None)


def extract_cftc_sp500_net(cftc_df: pd.DataFrame) -> pd.DataFrame:
    """Extract S&P 500 net non-commercial position from CFTC data.
    
    CFTC FinFut format (no header):
    - Column 0: Contract name
    - Column 2: Date (YYYY-MM-DD)
    - Column 8: Non-commercial Long
    - Column 9: Non-commercial Short
    Net = Long - Short
    """
    if cftc_df.empty:
        return pd.DataFrame(columns=["date", "CFTC_Net"])
    
    # Filter for S&P 500 Consolidated rows
    mask = cftc_df[0].str.contains("S&P 500 Consolidated", case=False, na=False)
    sp500 = cftc_df[mask].copy()
    
    if sp500.empty:
        return pd.DataFrame(columns=["date", "CFTC_Net"])
    
    # Extract date and positions
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(sp500[2], errors="coerce")
    
    # Columns 8 and 9 are non-commercial long and short
    long_col = pd.to_numeric(sp500[8], errors="coerce")
    short_col = pd.to_numeric(sp500[9], errors="coerce")
    result["CFTC_Net"] = long_col.values - short_col.values
    
    return result.dropna(subset=["date"]).sort_values("date")


def _detect_date_col(columns: list[str]) -> str | None:
    for c in columns:
        if "date" in str(c).lower():
            return c
    return None


def _detect_close_col(columns: list[str]) -> str | None:
    normalized = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in columns}
    for c, n in normalized.items():
        if "close" in n or "closing" in n:
            return c
    for c, n in normalized.items():
        if "nikkei" in n:
            return c
    return None


def fetch_nikkei_official_daily() -> pd.DataFrame:
    url = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_en.csv"
    df = fetch_csv(url)
    date_col = _detect_date_col(list(df.columns))
    close_col = _detect_close_col(list(df.columns))
    if not date_col or not close_col:
        raise RuntimeError(f"Nikkei CSV columns not recognized: {list(df.columns)}")
    out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    out["close"] = pd.to_numeric(
        out["close"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    return out


# ---- J-Quants (weekly margin balance) ----

def fetch_jquants_weekly_margin(api_key: str) -> pd.DataFrame:
    """Fetch weekly margin data from J-Quants V2 API."""
    url = "https://api.jquants.com/v2/markets/weekly_margin_interest"
    headers = {"x-api-key": api_key}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data.get("weekly_margin_interest", data.get("data", [])))


# ---- AAII ----

def fetch_aaii_from_mirror() -> pd.DataFrame:
    """Fetch AAII sentiment from official results page."""
    import time
    from io import StringIO

    url = "https://www.aaii.com/sentimentsurvey/sent_results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Retry with backoff
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            break
        except requests.HTTPError as e:
            if attempt < 2 and e.response.status_code in (503, 429):
                time.sleep(2 ** attempt)
                continue
            raise

    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        # Check if this looks like the AAII sentiment table
        vals = [str(v).lower() for v in t.values.flatten()[:10]]
        if any("bullish" in v for v in vals) and any("bearish" in v for v in vals):
            # First row is header
            df = t.copy()
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

            # Rename columns
            col_map = {}
            for c in df.columns:
                cl = str(c).lower()
                if "date" in cl:
                    col_map[c] = "date"
                elif "bull" in cl:
                    col_map[c] = "bullish"
                elif "neutral" in cl:
                    col_map[c] = "neutral"
                elif "bear" in cl:
                    col_map[c] = "bearish"
            df = df.rename(columns=col_map)

            # Parse date (format: "Jan 28" -> need to add year)
            if "date" in df.columns:
                current_year = datetime.now().year
                def parse_aaii_date(d: str) -> str:
                    try:
                        parsed = datetime.strptime(f"{d} {current_year}", "%b %d %Y")
                        if parsed > datetime.now():
                            parsed = datetime.strptime(f"{d} {current_year - 1}", "%b %d %Y")
                        return parsed.strftime("%Y-%m-%d")
                    except Exception:
                        return d
                df["date"] = df["date"].apply(parse_aaii_date)

            # Convert percentages to float
            for col in ["bullish", "neutral", "bearish"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("%", "").astype(float)

            return df

    raise RuntimeError("AAII table not found")


def fetch_aaii_from_manual_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)



def fetch_jpx_margin() -> pd.DataFrame:
    """Fetch margin trading balance from JPX website."""
    import io
    from datetime import timedelta

    base_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/tvdivq0000001rk9-att/mtseisan{date}00.xls"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    # Try recent dates (published every Wednesday)
    today = datetime.now()
    results = []

    for days_back in range(0, 30):
        check_date = today - timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")
        url = base_url.format(date=date_str)

        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                # Parse Excel
                df = pd.read_excel(io.BytesIO(r.content), header=None)

                # Find the row with "二市場計" (Total)
                for i, row in df.iterrows():
                    row_str = " ".join(str(v) for v in row.values if pd.notna(v))
                    if "二市場計" in row_str or "Total" in row_str:
                        # Next row has 株数 (shares) data
                        if i + 1 < len(df):
                            data_row = df.iloc[i]
                            # Find numeric columns (売残高, 買残高)
                            # Typically: col 3 = 売残高, col 5 = 買残高
                            try:
                                short_vol = pd.to_numeric(df.iloc[i, 3], errors="coerce")
                                long_vol = pd.to_numeric(df.iloc[i, 5], errors="coerce")
                                if pd.notna(short_vol) and pd.notna(long_vol):
                                    results.append({
                                        "date": check_date.strftime("%Y-%m-%d"),
                                        "margin_long": int(long_vol),
                                        "margin_short": int(short_vol),
                                        "margin_balance": int(long_vol - short_vol),
                                    })
                                    break
                            except Exception:
                                pass
                if results:
                    break
        except Exception:
            continue

    if not results:
        raise RuntimeError("JPX margin data not found")

    return pd.DataFrame(results)

def fetch_aaii(mode: str, manual_path: str) -> tuple[pd.DataFrame, str]:
    mode = (mode or "mirror").lower()
    if mode == "mirror":
        return fetch_aaii_from_mirror(), "mirror"
    if mode == "manual":
        return fetch_aaii_from_manual_excel(manual_path), "manual"
    if mode in ("auto", "both"):
        errors: list[str] = []
        try:
            return fetch_aaii_from_mirror(), "mirror"
        except Exception as e:
            errors.append(f"mirror:{e.__class__.__name__}")
        try:
            return fetch_aaii_from_manual_excel(manual_path), "manual"
        except Exception as e:
            errors.append(f"manual:{e.__class__.__name__}")
        raise RuntimeError("AAII fetch failed (" + ", ".join(errors) + ")")
    raise RuntimeError("AAII_MODE must be mirror/manual/auto")


# -----------------------
# Main
# -----------------------

def _safe_symbol(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sym).strip("_")


def main() -> None:
    ensure_dirs()

    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    aaii_mode = os.environ.get("AAII_MODE", "mirror")
    aaii_manual_file = os.environ.get("AAII_MANUAL_FILE", os.path.join(INPUT_DIR, "aaii.xlsx"))
    # GitHub Pages base URL for chart images
    chart_base_url = os.environ.get("CHART_BASE_URL", "")

    default_targets = "^spx,NIKKEI_OFFICIAL,fx.f,acwi.us"
    targets = [s.strip() for s in os.environ.get("RSI_TARGETS", default_targets).split(",") if s.strip()]

    # Store DataFrames for combined output
    vix_df: pd.DataFrame | None = None
    aaii_df: pd.DataFrame | None = None
    cftc_df: pd.DataFrame | None = None
    margin_df: pd.DataFrame | None = None
    rsi_df: pd.DataFrame | None = None

    # ---- VIX ----
    vix_df = fetch_vix_fred()
    save_csv(vix_df, os.path.join(DATA_DIR, "vix.csv"))
    save_line_chart(vix_df, "date", "vix", "VIX (FRED VIXCLS)", os.path.join(CHART_DIR, "vix.png"))
    vix_dates = pd.to_datetime(vix_df["date"], errors="coerce")
    vix_max = vix_dates.max()
    vix_key = str(vix_max.date()) if pd.notna(vix_max) else "unknown"
    vix_updated = changed_since_last_run("vix", vix_key)

    # ---- CFTC ----
    cftc_df = fetch_cftc_finfutwk()
    save_csv(cftc_df, os.path.join(DATA_DIR, "cftc_finfutwk_raw.csv"))
    cftc_key = str(len(cftc_df))
    cftc_updated = changed_since_last_run("cftc", cftc_key)

    # ---- J-Quants weekly margin ----
    margin_updated = False
    margin_note = "信用残: -"
    try:
        margin_df = fetch_jpx_margin()
        save_csv(margin_df, os.path.join(DATA_DIR, "margin_jpx.csv"))
        if "date" in margin_df.columns:
            key = str(margin_df["date"].iloc[0])
            margin_updated = changed_since_last_run("margin", key)
            margin_note = f"信用残: {'更新あり' if margin_updated else '更新なし'} ({key})"
    except Exception as e:
        margin_note = f"信用残: 取得失敗 ({e.__class__.__name__})"

    # ---- AAII ----
    aaii_updated = False
    aaii_note = "AAII: 未取得"
    try:
        aaii_df, aaii_source = fetch_aaii(aaii_mode, aaii_manual_file)
        save_csv(aaii_df, os.path.join(DATA_DIR, "aaii.csv"))
        key = str(len(aaii_df))
        for cand in aaii_df.columns:
            if "date" in str(cand).lower():
                max_date = pd.to_datetime(aaii_df[cand], errors="coerce").max()
                key = str(max_date.date()) if pd.notna(max_date) else "unknown"
                break
        aaii_updated = changed_since_last_run("aaii", key)
        aaii_note = f"AAII({aaii_source}): {'更新あり' if aaii_updated else '更新なし'} ({key})"
    except Exception as e:
        aaii_note = f"AAII: 取得失敗 ({e.__class__.__name__})"

    # ---- RSI ----
    rsi_lines: list[str] = []
    primary_rsi_df: pd.DataFrame | None = None
    for sym in targets:
        try:
            if sym.upper() == NIKKEI_OFFICIAL_SYMBOL:
                px = fetch_nikkei_official_daily()
                label = "Nikkei 225 (Official)"
            else:
                px = fetch_stooq(sym)
                px = px.rename(columns={"Date": "date", "Close": "close"})
                label = sym

            px = px[["date", "close"]].copy()
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px["close"] = pd.to_numeric(
                px["close"].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
            px = px.dropna(subset=["date", "close"]).sort_values("date")
            px["RSI14"] = rsi(px["close"], 14)

            safe_sym = _safe_symbol(sym)
            out_csv = os.path.join(DATA_DIR, f"{safe_sym}_rsi.csv")
            out_png = os.path.join(CHART_DIR, f"rsi_{safe_sym}.png")
            save_csv(px, out_csv)
            save_line_chart(px, "date", "RSI14", f"RSI14 {label}", out_png)

            # Use first target (^spx) as primary RSI for combined chart
            if primary_rsi_df is None:
                primary_rsi_df = px[["date", "RSI14"]].copy()

            last = px.dropna(subset=["RSI14"]).iloc[-1]
            rsi_lines.append(f"RSI {label}: {float(last['RSI14']):.1f}")
        except Exception as e:
            rsi_lines.append(f"RSI {sym}: 失敗({e.__class__.__name__})")

    rsi_df = primary_rsi_df

    # ---- Combined output ----
    combined = build_combined_df(vix_df, aaii_df, cftc_df, margin_df, rsi_df)
    if not combined.empty:
        save_csv(combined, os.path.join(DATA_DIR, "combined.csv"))
        save_combined_chart(combined, os.path.join(CHART_DIR, "combined.png"))

    # ---- Slack ----
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(timezone.utc).astimezone(jst).strftime("%Y-%m-%d %H:%M JST")
    lines = [
        f"Daily Market Reminder ({now_jst})",
        f"VIX: {'更新あり' if vix_updated else '更新なし'} ({vix_key})",
        aaii_note,
        f"CFTC: {'更新あり' if cftc_updated else '更新なし'}",
        margin_note,
        " / ".join(rsi_lines) if rsi_lines else "RSI: なし",
        "",
        "出力: data/*.csv, charts/*.png",
    ]
    message = "\n".join(lines)
    print(message)

    # Build image URLs if base URL is configured
    image_urls: list[str] = []
    if chart_base_url:
        image_urls = [
            f"{chart_base_url.rstrip('/')}/charts/combined.png",
        ]

    slack_notify(slack_webhook, message, image_urls if image_urls else None)


def notify_only() -> None:
    """Send Slack notification using saved state (for use after commit)."""
    import json

    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    chart_base_url = os.environ.get("CHART_BASE_URL", "")

    if not slack_webhook:
        print("SLACK_WEBHOOK_URL not set, skipping notification")
        return

    # Load saved state
    state_file = os.path.join(DATA_DIR, "_state.json")
    state: dict[str, str] = {}
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)

    # Build message from state
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(timezone.utc).astimezone(jst).strftime("%m/%d %H:%M")

    # Read latest values from data files
    vix_val = 0.0
    aaii_bull = 0.0
    aaii_bear = 0.0
    cftc_net = 0
    margin_balance = 0
    rsi_val = 50.0

    # VIX
    vix_path = os.path.join(DATA_DIR, "vix.csv")
    if os.path.exists(vix_path):
        try:
            df = pd.read_csv(vix_path)
            vix_val = float(df.dropna(subset=["vix"]).iloc[-1]["vix"])
        except Exception:
            pass

    # AAII
    aaii_path = os.path.join(DATA_DIR, "aaii.csv")
    if os.path.exists(aaii_path):
        try:
            df = pd.read_csv(aaii_path)
            last = df.iloc[0]  # Latest is first row
            aaii_bull = float(last.get("bullish", 0))
            aaii_bear = float(last.get("bearish", 0))
        except Exception:
            pass

    # CFTC
    combined_path = os.path.join(DATA_DIR, "combined.csv")
    if os.path.exists(combined_path):
        try:
            df = pd.read_csv(combined_path)
            df = df.dropna(subset=["date"]).sort_values("date")
            if "CFTC_Net" in df.columns and df["CFTC_Net"].notna().any():
                cftc_net = int(df.dropna(subset=["CFTC_Net"]).iloc[-1]["CFTC_Net"])
        except Exception:
            pass

    # Margin
    margin_path = os.path.join(DATA_DIR, "margin_jpx.csv")
    if os.path.exists(margin_path):
        try:
            df = pd.read_csv(margin_path)
            margin_balance = int(df.iloc[0].get("margin_balance", 0))
        except Exception:
            pass

    # Read RSI from CSV if not in combined
    if rsi_val == 50.0:
        csv_path = os.path.join(DATA_DIR, "spx_rsi.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                rsi_val = float(df.dropna(subset=["RSI14"]).iloc[-1]["RSI14"])
            except Exception:
                pass

    # Weather-like indicators
    def get_vix_weather(vix: float) -> str:
        if vix < 15:
            return "☀️ 快晴"
        elif vix < 20:
            return "🌤️ 晴れ"
        elif vix < 25:
            return "⛅ くもり"
        elif vix < 30:
            return "🌧️ 雨"
        elif vix < 40:
            return "⛈️ 荒れ模様"
        else:
            return "🌪️ 暴風雨"

    def get_sentiment_weather(bull: float, bear: float) -> str:
        spread = bull - bear
        if spread > 20:
            return "😊 楽観"
        elif spread > 10:
            return "🙂 やや楽観"
        elif spread > -10:
            return "😐 中立"
        elif spread > -20:
            return "😟 やや悲観"
        else:
            return "😰 悲観"

    def get_rsi_indicator(rsi: float) -> str:
        if rsi >= 70:
            return "🔥 過熱"
        elif rsi >= 60:
            return "🌡️ やや過熱"
        elif rsi <= 30:
            return "❄️ 売られすぎ"
        elif rsi <= 40:
            return "🌬️ やや弱い"
        else:
            return "🌡️ 適温"

    def get_external_pressure(cftc: int) -> str:
        if cftc < -150000:
            return "💨💨💨 外圧強い"
        elif cftc < -100000:
            return "💨💨 外圧やや強い"
        elif cftc < -50000:
            return "💨 外圧あり"
        elif cftc > 50000:
            return "🌬️ 買い越し"
        else:
            return "〜 穏やか"

    vix_weather = get_vix_weather(vix_val)
    sentiment = get_sentiment_weather(aaii_bull, aaii_bear)
    rsi_indicator = get_rsi_indicator(rsi_val)
    pressure = get_external_pressure(cftc_net)

    lines = [
        "<!here>",
        f"📊 マーケット天気予報 ({now_jst})",
        "",
        f"🌡️ VIX: {vix_val:.1f} → {vix_weather}",
        f"👥 センチメント: {sentiment} (強気{aaii_bull:.0f}%/弱気{aaii_bear:.0f}%)",
        f"📈 RSI: {rsi_val:.1f} → {rsi_indicator}",
        f"🌊 CFTC: {cftc_net:,} → {pressure}",
    ]

    if margin_balance > 0:
        margin_ratio = margin_balance / 1000000  # 百万株単位
        lines.append(f"💰 信用残: {margin_ratio:.1f}M株")

    message = "\n".join(lines)
    print(message)

    # Build image URLs
    image_urls: list[str] = []
    if chart_base_url:
        image_urls = [
            f"{chart_base_url.rstrip('/')}/charts/combined.png",
        ]

    slack_notify(slack_webhook, message, image_urls if image_urls else None)


if __name__ == "__main__":
    import sys
    if "--notify-only" in sys.argv:
        notify_only()
    else:
        main()
