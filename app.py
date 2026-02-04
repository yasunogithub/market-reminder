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

def fetch_jquants_weekly_margin(token: str) -> pd.DataFrame:
    api_url = os.environ.get("JQUANTS_MARGIN_API_URL", "")
    if not api_url:
        raise RuntimeError("JQUANTS_MARGIN_API_URL is not set")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(api_url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data.get("data", data))


# ---- AAII ----

def fetch_aaii_from_mirror() -> pd.DataFrame:
    url = "https://ibmetrics.com/aaii-sentiment.html"
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("bull" in c for c in cols) and any("bear" in c for c in cols):
            return t.copy()
    raise RuntimeError("AAII mirror table not found")


def fetch_aaii_from_manual_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)


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
    margin_note = "信用残: 未設定"
    j_token = os.environ.get("JQUANTS_TOKEN", "")
    if j_token and os.environ.get("JQUANTS_MARGIN_API_URL", ""):
        try:
            margin_df = fetch_jquants_weekly_margin(j_token)
            save_csv(margin_df, os.path.join(DATA_DIR, "margin_weekly.csv"))
            key = str(len(margin_df))
            for cand in ["Date", "date", "EndDate", "end_date", "TradeDate", "trade_date"]:
                if cand in margin_df.columns:
                    max_date = pd.to_datetime(margin_df[cand], errors="coerce").max()
                    key = str(max_date.date()) if pd.notna(max_date) else "unknown"
                    break
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
    now_jst = datetime.now(timezone.utc).astimezone(jst).strftime("%Y-%m-%d %H:%M JST")

    vix_key = state.get("vix", "unknown")

    # Read RSI values from saved CSVs
    rsi_lines: list[str] = []
    default_targets = "^spx,NIKKEI_OFFICIAL,fx.f,acwi.us"
    targets = [s.strip() for s in os.environ.get("RSI_TARGETS", default_targets).split(",") if s.strip()]
    for sym in targets:
        safe_sym = _safe_symbol(sym)
        csv_path = os.path.join(DATA_DIR, f"{safe_sym}_rsi.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                last = df.dropna(subset=["RSI14"]).iloc[-1]
                label = "Nikkei 225 (Official)" if sym.upper() == NIKKEI_OFFICIAL_SYMBOL else sym
                rsi_lines.append(f"RSI {label}: {float(last['RSI14']):.1f}")
            except Exception:
                pass

    # Read latest values from combined.csv
    combined_path = os.path.join(DATA_DIR, "combined.csv")
    vix_val = "-"
    cftc_val = "-"
    if os.path.exists(combined_path):
        try:
            df = pd.read_csv(combined_path)
            df = df.dropna(subset=["date"]).sort_values("date")
            if "vix" in df.columns:
                last_vix = df.dropna(subset=["vix"]).iloc[-1] if df["vix"].notna().any() else None
                if last_vix is not None:
                    vix_val = f"{float(last_vix['vix']):.2f}"
            if "CFTC_Net" in df.columns:
                last_cftc = df.dropna(subset=["CFTC_Net"]).iloc[-1] if df["CFTC_Net"].notna().any() else None
                if last_cftc is not None:
                    cftc_val = f"{int(last_cftc['CFTC_Net']):,}"
        except Exception:
            pass

    lines = [
        f"Daily Market Reminder ({now_jst})",
        f"VIX: {vix_val} ({vix_key})",
        " / ".join(rsi_lines) if rsi_lines else "RSI: -",
        f"CFTC Net: {cftc_val}",
    ]
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
