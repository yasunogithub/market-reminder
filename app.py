import io
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import pdfplumber
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
    """Fetch margin trading balance from JPX website (incremental update)."""
    import io
    from datetime import timedelta

    base_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/tvdivq0000001rk9-att/mtseisan{date}00.xls"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    # Load existing data if available
    existing_df = None
    existing_dates = set()
    csv_path = os.path.join(DATA_DIR, "margin_jpx.csv")
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            existing_dates = set(existing_df["date"].astype(str).tolist())
        except Exception:
            pass

    today = datetime.now()
    results = []
    checked_dates = set()

    # Only fetch recent 2 weeks (data is published weekly on Wednesdays)
    for weeks_back in range(0, 2):
        base_date = today - timedelta(weeks=weeks_back)
        
        # Try Wednesday and nearby days
        for day_offset in range(-2, 3):
            check_date = base_date + timedelta(days=day_offset)
            date_str = check_date.strftime("%Y%m%d")
            date_key = check_date.strftime("%Y-%m-%d")
            
            # Skip if already in existing data or already checked
            if date_key in existing_dates or date_str in checked_dates:
                continue
            checked_dates.add(date_str)

            url = base_url.format(date=date_str)

            try:
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code != 200:
                    continue

                # Parse Excel
                df = pd.read_excel(io.BytesIO(r.content), header=None)

                # Find the row with "二市場計" (Total)
                for i, row in df.iterrows():
                    row_str = " ".join(str(v) for v in row.values if pd.notna(v))
                    if "二市場計" in row_str or "Total" in row_str:
                        try:
                            short_vol = pd.to_numeric(df.iloc[i, 3], errors="coerce")
                            long_vol = pd.to_numeric(df.iloc[i, 5], errors="coerce")
                            if pd.notna(short_vol) and pd.notna(long_vol):
                                results.append({
                                    "date": date_key,
                                    "margin_long": int(long_vol),
                                    "margin_short": int(short_vol),
                                    "margin_balance": int(long_vol - short_vol),
                                })
                                break
                        except Exception:
                            pass
            except Exception:
                continue

    # Merge with existing data
    if results:
        new_df = pd.DataFrame(results)
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
    elif existing_df is not None:
        combined = existing_df
    else:
        raise RuntimeError("JPX margin data not found")

    # Remove duplicates and sort
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date", ascending=False)
    return combined

def fetch_oil_stockpile() -> pd.DataFrame:
    """Fetch Japan oil stockpile data (speed report) from METI/ENECHO.

    Scrapes the statistics page to find the latest speed-report PDF,
    downloads it, and extracts stockpile days for each category.
    Returns a DataFrame with columns:
        date, national, private, joint, total
    """
    base = "https://www.enecho.meti.go.jp"
    index_url = base + "/statistics/petroleum_and_lpgas/pl001/"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(index_url, headers=headers, timeout=30)
    r.raise_for_status()
    m = re.search(r'href=["\']([^"\']*pdf-oil-res/[^"\']+\.pdf)', r.text)
    if not m:
        raise RuntimeError("Oil stockpile speed-report PDF link not found")
    pdf_path = m.group(1)
    if not pdf_path.startswith("http"):
        pdf_path = base + pdf_path

    pr = requests.get(pdf_path, headers=headers, timeout=30)
    pr.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pr.content)
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Normalize full-width digits to half-width
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    rows: list[dict] = []
    # Split into blocks by date header: 令和N年M月D日（M月D日時点）
    blocks = re.split(r"(令和\d+年\d+月\s*\d+日)", text)
    for i in range(1, len(blocks), 2):
        date_header = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""

        # Parse date from header (令和8年4月3日 -> 2026-04-03)
        dm = re.search(r"令和(\d+)年(\d+)月\s*(\d+)日", date_header)
        if not dm:
            continue
        year = int(dm.group(1)) + 2018
        month = int(dm.group(2))
        day = int(dm.group(3))
        date_str = f"{year}-{month:02d}-{day:02d}"

        # Extract stockpile days
        national = _extract_days(body, r"国家備蓄\s*(\d+)")
        private = _extract_days(body, r"民間備蓄\s*(\d+)")
        joint = _extract_days(body, r"産油国共同備蓄\s*(\d+)")
        total = _extract_days(body, r"合計\s*(\d+)")
        if total is None:
            continue

        rows.append({
            "date": date_str,
            "national": national,
            "private": private,
            "joint": joint,
            "total": total,
        })

    if not rows:
        raise RuntimeError("Failed to parse oil stockpile data from PDF")

    df = pd.DataFrame(rows)
    # Merge with existing data if present
    csv_path = os.path.join(DATA_DIR, "oil_stockpile.csv")
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, dtype=str)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


def _extract_days(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


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

    # Regional RSI targets with labels
    # Format: symbol or symbol:label
    default_targets = "^spx:🇺🇸S&P500,NIKKEI_OFFICIAL:🇯🇵日経225,^dax:🇪🇺DAX,^hsi:🇨🇳ハンセン,^rts:🇷🇺RTS,acwi.us:🌍ACWI"
    targets_str = os.environ.get("RSI_TARGETS") or default_targets  # Handle empty string
    targets = [s.strip() for s in targets_str.split(",") if s.strip()]

    # Store DataFrames for combined output
    vix_df: pd.DataFrame | None = None
    aaii_df: pd.DataFrame | None = None
    cftc_df: pd.DataFrame | None = None
    margin_df: pd.DataFrame | None = None
    rsi_df: pd.DataFrame | None = None
    oil_stockpile_df: pd.DataFrame | None = None

    # ---- Parallel data fetching ----
    def fetch_task_vix():
        return ("vix", fetch_vix_fred())

    def fetch_task_cftc():
        return ("cftc", fetch_cftc_finfutwk())

    def fetch_task_margin():
        try:
            return ("margin", fetch_jpx_margin())
        except Exception as e:
            return ("margin", e)

    def fetch_task_aaii():
        try:
            return ("aaii", fetch_aaii(aaii_mode, aaii_manual_file))
        except Exception as e:
            return ("aaii", e)

    def fetch_task_oil_stockpile():
        try:
            return ("oil_stockpile", fetch_oil_stockpile())
        except Exception as e:
            return ("oil_stockpile", e)

    def fetch_task_rsi(target: str):
        if ":" in target:
            sym, label = target.split(":", 1)
        else:
            sym, label = target, target
        try:
            if sym.upper() == NIKKEI_OFFICIAL_SYMBOL or "NIKKEI" in sym.upper():
                px = fetch_nikkei_official_daily()
            else:
                px = fetch_stooq(sym)
                px = px.rename(columns={"Date": "date", "Close": "close"})
            return ("rsi", sym, label, px)
        except Exception as e:
            return ("rsi", sym, label, e)

    # Run all fetches in parallel
    results = {}
    rsi_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fetch_task_vix),
            executor.submit(fetch_task_cftc),
            executor.submit(fetch_task_margin),
            executor.submit(fetch_task_aaii),
            executor.submit(fetch_task_oil_stockpile),
        ]
        # Add RSI targets
        for t in targets:
            futures.append(executor.submit(fetch_task_rsi, t))

        for future in as_completed(futures):
            result = future.result()
            if result[0] == "rsi":
                rsi_results.append(result)
            else:
                results[result[0]] = result[1]

    # ---- Process VIX ----
    vix_df = results.get("vix")
    if vix_df is not None:
        save_csv(vix_df, os.path.join(DATA_DIR, "vix.csv"))
        save_line_chart(vix_df, "date", "vix", "VIX (FRED VIXCLS)", os.path.join(CHART_DIR, "vix.png"))
    vix_dates = pd.to_datetime(vix_df["date"], errors="coerce") if vix_df is not None else pd.Series()
    vix_max = vix_dates.max() if not vix_dates.empty else None
    vix_key = str(vix_max.date()) if pd.notna(vix_max) else "unknown"
    vix_updated = changed_since_last_run("vix", vix_key)

    # ---- Process CFTC ----
    cftc_df = results.get("cftc")
    if cftc_df is not None:
        save_csv(cftc_df, os.path.join(DATA_DIR, "cftc_finfutwk_raw.csv"))
    cftc_key = str(len(cftc_df)) if cftc_df is not None else "0"
    cftc_updated = changed_since_last_run("cftc", cftc_key)

    # ---- Process Margin ----
    margin_updated = False
    margin_note = "信用残: -"
    margin_result = results.get("margin")
    if isinstance(margin_result, pd.DataFrame):
        margin_df = margin_result
        save_csv(margin_df, os.path.join(DATA_DIR, "margin_jpx.csv"))
        if "date" in margin_df.columns:
            key = str(margin_df["date"].iloc[0])
            margin_updated = changed_since_last_run("margin", key)
            margin_note = f"信用残: {'更新あり' if margin_updated else '更新なし'} ({key})"
    elif isinstance(margin_result, Exception):
        margin_note = f"信用残: 取得失敗 ({margin_result.__class__.__name__})"

    # ---- Process AAII ----
    aaii_updated = False
    aaii_note = "AAII: 未取得"
    aaii_result = results.get("aaii")
    if isinstance(aaii_result, tuple):
        aaii_df, aaii_source = aaii_result
        save_csv(aaii_df, os.path.join(DATA_DIR, "aaii.csv"))
        key = str(len(aaii_df))
        for cand in aaii_df.columns:
            if "date" in str(cand).lower():
                max_date = pd.to_datetime(aaii_df[cand], errors="coerce").max()
                key = str(max_date.date()) if pd.notna(max_date) else "unknown"
                break
        aaii_updated = changed_since_last_run("aaii", key)
        aaii_note = f"AAII({aaii_source}): {'更新あり' if aaii_updated else '更新なし'} ({key})"
    elif isinstance(aaii_result, Exception):
        aaii_note = f"AAII: 取得失敗 ({aaii_result.__class__.__name__})"

    # ---- Process Oil Stockpile ----
    oil_updated = False
    oil_note = "石油備蓄: 未取得"
    oil_result = results.get("oil_stockpile")
    if isinstance(oil_result, pd.DataFrame):
        oil_stockpile_df = oil_result
        save_csv(oil_stockpile_df, os.path.join(DATA_DIR, "oil_stockpile.csv"))
        save_line_chart(
            oil_stockpile_df, "date", "total",
            "Japan Oil Stockpile (days)",
            os.path.join(CHART_DIR, "oil_stockpile.png"),
        )
        latest = oil_stockpile_df.iloc[0]
        key = str(latest["date"])[:10]
        oil_updated = changed_since_last_run("oil_stockpile", key)
        oil_note = (
            f"石油備蓄: {'更新あり' if oil_updated else '更新なし'} ({key}) "
            f"合計{int(latest['total'])}日 "
            f"(国家{int(latest['national'])} / 民間{int(latest['private'])} / 共同{int(latest['joint'])})"
        )
    elif isinstance(oil_result, Exception):
        oil_note = f"石油備蓄: 取得失敗 ({oil_result.__class__.__name__})"

    # ---- Process RSI (maintain order) ----
    rsi_lines: list[str] = []
    primary_rsi_df: pd.DataFrame | None = None
    # Sort by original target order
    target_order = {t.split(":")[0] if ":" in t else t: i for i, t in enumerate(targets)}
    rsi_results.sort(key=lambda x: target_order.get(x[1], 999))

    for result in rsi_results:
        _, sym, label, data = result
        if isinstance(data, Exception):
            rsi_lines.append(f"{label}: 失敗({data.__class__.__name__})")
            continue

        try:
            px = data
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
            rsi_lines.append(f"{label}: {float(last['RSI14']):.1f}")
        except Exception as e:
            rsi_lines.append(f"{label}: 失敗({e.__class__.__name__})")

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
        oil_note,
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



def generate_market_insights(
    vix: float, rsi: float, cftc: int, bull: float, bear: float, margin: int
) -> str:
    """Generate detailed market analysis insights."""
    insights = ["🤖 *AI Market Insights*", ""]
    
    # VIX analysis
    if vix < 15:
        insights.append("📊 *VIX分析*: 極めて低水準。市場は安心感に包まれていますが、")
        insights.append("   コンプレイセンシー(油断)のサインかも。急変に注意。")
    elif vix < 20:
        insights.append("📊 *VIX分析*: 安定した水準。通常の相場環境です。")
    elif vix < 30:
        insights.append("📊 *VIX分析*: やや警戒水準。不安定な動きに備えを。")
    else:
        insights.append("📊 *VIX分析*: 高水準。恐怖が市場を支配中。")
        insights.append("   逆張り的には買いチャンスの可能性も。")
    
    # RSI analysis
    insights.append("")
    if rsi <= 30:
        insights.append("📈 *RSI分析*: 売られすぎ水準。反発のタイミングを探る局面。")
        insights.append("   ただし、トレンドが強い場合はさらに下落も。")
    elif rsi >= 70:
        insights.append("📈 *RSI分析*: 買われすぎ水準。利確タイミングを検討。")
        insights.append("   強いトレンドでは高止まりすることも。")
    elif rsi >= 60:
        insights.append("📈 *RSI分析*: やや過熱気味。上昇トレンドは継続中。")
    elif rsi <= 40:
        insights.append("📈 *RSI分析*: やや弱含み。下落トレンドに注意。")
    else:
        insights.append("📈 *RSI分析*: 中立水準。方向感を探る展開。")
    
    # CFTC analysis
    insights.append("")
    if cftc < -150000:
        insights.append("🌊 *CFTC分析*: 投機筋の大幅売り越し。外国人売り圧力が強い。")
        insights.append("   売り一巡後の反発に期待も、需給悪化に警戒。")
    elif cftc < -50000:
        insights.append("🌊 *CFTC分析*: 投機筋は売り越し基調。上値重い展開か。")
    elif cftc > 50000:
        insights.append("🌊 *CFTC分析*: 投機筋は買い越し。強気姿勢維持。")
    else:
        insights.append("🌊 *CFTC分析*: 投機筋のポジションは中立的。")
    
    # Sentiment analysis
    insights.append("")
    spread = bull - bear
    if spread < -20:
        insights.append("👥 *センチメント分析*: 極端な悲観。歴史的にはここが底になりやすい。")
        insights.append("   「皆が売りたい時が買い時」の格言通りか。")
    elif spread > 30:
        insights.append("👥 *センチメント分析*: 過度な楽観。警戒が必要な水準。")
        insights.append("   「皆が買いたい時が売り時」かもしれません。")
    else:
        insights.append("👥 *センチメント分析*: 極端なポジションなし。通常の心理状態。")
    
    # Margin analysis
    if margin > 0:
        insights.append("")
        margin_m = margin / 1000000
        if margin_m > 4:
            insights.append(f"💰 *信用残分析*: {margin_m:.1f}M株は高水準。将来の売り圧力に。")
            insights.append("   整理売りが出やすい環境。")
        elif margin_m < 2:
            insights.append(f"💰 *信用残分析*: {margin_m:.1f}M株は低水準。売り圧力は限定的。")
            insights.append("   信用買い余力あり。")
        else:
            insights.append(f"💰 *信用残分析*: {margin_m:.1f}M株は通常水準。")
    
    return "\n".join(insights)

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

    # Read regional RSI values
    regional_rsi: dict[str, float] = {}
    rsi_files = {
        "🇺🇸米国": "spx_rsi.csv",
        "🇯🇵日本": "NIKKEI_OFFICIAL_rsi.csv",
        "🇪🇺欧州": "dax_rsi.csv",
        "🇨🇳中国": "hsi_rsi.csv",
        "🇷🇺ロシア": "rts_rsi.csv",
        "🌍世界": "acwi.us_rsi.csv",
    }
    for region, filename in rsi_files.items():
        csv_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                regional_rsi[region] = float(df.dropna(subset=["RSI14"]).iloc[-1]["RSI14"])
            except Exception:
                pass

    # Read USD/JPY for Japan analysis
    usdjpy = 0.0
    try:
        usdjpy_df = fetch_stooq("usdjpy")
        if not usdjpy_df.empty and "Close" in usdjpy_df.columns:
            usdjpy = float(usdjpy_df.iloc[-1]["Close"])
    except Exception:
        pass

    # Weather-like indicators

    # Trend analysis for forecasting
    def get_vix_trend(vix_path: str) -> tuple[str, float]:
        """Analyze VIX trend over last 5 days."""
        try:
            df = pd.read_csv(vix_path)
            df = df.dropna(subset=["vix"]).tail(5)
            if len(df) < 2:
                return "→", 0.0
            change = df["vix"].iloc[-1] - df["vix"].iloc[0]
            pct = (change / df["vix"].iloc[0]) * 100
            if pct > 10:
                return "↑↑", pct
            elif pct > 3:
                return "↑", pct
            elif pct < -10:
                return "↓↓", pct
            elif pct < -3:
                return "↓", pct
            else:
                return "→", pct
        except Exception:
            return "→", 0.0

    def get_rsi_trend(csv_path: str) -> tuple[str, float]:
        """Analyze RSI trend over last 5 days."""
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=["RSI14"]).tail(5)
            if len(df) < 2:
                return "→", 0.0
            change = df["RSI14"].iloc[-1] - df["RSI14"].iloc[0]
            if change > 5:
                return "↑", change
            elif change < -5:
                return "↓", change
            else:
                return "→", change
        except Exception:
            return "→", 0.0

    def generate_tomorrow_outlook(
        vix: float, vix_trend: str, rsi: float, rsi_trend: str,
        cftc: int, bull: float, bear: float
    ) -> str:
        """Generate tomorrow's market outlook."""
        signals = []
        score = 0

        # VIX factor with trend
        if vix < 20 and vix_trend in ("↓", "↓↓"):
            signals.append("VIX低下中")
            score += 1
        elif vix > 25 and vix_trend in ("↑", "↑↑"):
            signals.append("VIX上昇警戒")
            score -= 1

        # RSI factor with trend
        if rsi < 40 and rsi_trend == "↑":
            signals.append("売られすぎから反発")
            score += 1
        elif rsi > 60 and rsi_trend == "↓":
            signals.append("過熱感解消")
            score += 0  # neutral
        elif rsi > 65 and rsi_trend in ("↑", "→"):
            signals.append("過熱継続")
            score -= 1

        # Sentiment extremes (contrarian short-term signals)
        spread = bull - bear
        if spread < -15:
            signals.append("悲観反発期待")
            score += 1
        elif spread > 25:
            signals.append("楽観警戒")
            score -= 1

        if score >= 1:
            outlook = "🌤️ 上向き"
        elif score <= -1:
            outlook = "🌧️ 軟調"
        else:
            outlook = "⛅ 横ばい"

        return outlook, signals

    def generate_weekly_outlook(
        vix: float, vix_trend: str, rsi: float, cftc: int,
        bull: float, bear: float, margin: int
    ) -> str:
        """Generate weekly market outlook."""
        factors = []
        score = 0

        # VIX level (mean reversion)
        if vix < 15:
            factors.append("低VIX継続リスク")
            score -= 1  # complacency risk
        elif vix > 30:
            factors.append("恐怖からの回復期待")
            score += 1  # fear reversal

        # RSI position
        if rsi < 35:
            factors.append("底値圏")
            score += 1
        elif rsi > 70:
            factors.append("天井圏")
            score -= 1

        # CFTC positioning (contrarian for weekly)
        if cftc < -100000:
            factors.append("投機筋売り一巡期待")
            score += 1
        elif cftc > 100000:
            factors.append("投機筋ロング過多")
            score -= 1

        # Sentiment (contrarian)
        spread = bull - bear
        if spread < -20:
            factors.append("極端な悲観→反転候補")
            score += 1
        elif spread > 30:
            factors.append("過度な楽観→調整候補")
            score -= 1

        # Margin pressure
        if margin > 0:
            margin_m = margin / 1000000
            if margin_m > 4:
                factors.append("信用売り圧力あり")
                score -= 1

        if score >= 2:
            outlook = "📈 上昇基調"
        elif score >= 1:
            outlook = "🌤️ やや強気"
        elif score <= -2:
            outlook = "📉 下落警戒"
        elif score <= -1:
            outlook = "🌧️ やや弱気"
        else:
            outlook = "➡️ レンジ推移"

        return outlook, factors

    # Get trends
    vix_trend, vix_change = get_vix_trend(vix_path)
    rsi_trend, rsi_change = get_rsi_trend(os.path.join(DATA_DIR, "spx_rsi.csv"))

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

    # Generate overall outlook
    def get_outlook(vix: float, rsi: float, cftc: int, bull: float, bear: float) -> str:
        score = 0
        reasons = []
        
        # VIX factor
        if vix < 15:
            score += 2
        elif vix < 20:
            score += 1
        elif vix > 30:
            score -= 2
            reasons.append("VIX高水準")
        elif vix > 25:
            score -= 1
        
        # RSI factor
        if rsi <= 30:
            score += 2
            reasons.append("売られすぎ水準")
        elif rsi <= 40:
            score += 1
        elif rsi >= 70:
            score -= 2
            reasons.append("過熱感")
        elif rsi >= 60:
            score -= 1
        
        # CFTC factor
        if cftc < -100000:
            score -= 1
            reasons.append("海外勢売り越し")
        elif cftc > 50000:
            score += 1
        
        # Sentiment factor (contrarian)
        spread = bull - bear
        if spread < -20:
            score += 1
            reasons.append("悲観極まり")
        elif spread > 30:
            score -= 1
            reasons.append("楽観過ぎ")
        
        if score >= 2:
            outlook = "📈 上昇期待"
        elif score >= 1:
            outlook = "↗️ やや上目線"
        elif score <= -2:
            outlook = "📉 下落警戒"
        elif score <= -1:
            outlook = "↘️ やや下目線"
        else:
            outlook = "➡️ 様子見"
        
        return outlook, reasons

    outlook, reasons = get_outlook(vix_val, rsi_val, cftc_net, aaii_bull, aaii_bear)

    # Generate tomorrow and weekly outlooks
    tomorrow_outlook, tomorrow_signals = generate_tomorrow_outlook(
        vix_val, vix_trend, rsi_val, rsi_trend, cftc_net, aaii_bull, aaii_bear
    )
    weekly_outlook, weekly_factors = generate_weekly_outlook(
        vix_val, vix_trend, rsi_val, cftc_net, aaii_bull, aaii_bear, margin_balance
    )

    lines = [
        "<!here>",
        f"📊 マーケット天気予報 ({now_jst})",
        "",
        f"🎯 総合判断: {outlook}",
    ]
    
    if reasons:
        lines.append(f"   └ {', '.join(reasons)}")

    # Add tomorrow and weekly outlook
    lines.extend([
        "",
        f"📅 *明日の見通し*: {tomorrow_outlook}",
    ])
    if tomorrow_signals:
        lines.append(f"   └ {', '.join(tomorrow_signals)}")

    lines.extend([
        f"📆 *今後1週間*: {weekly_outlook}",
    ])
    if weekly_factors:
        lines.append(f"   └ {', '.join(weekly_factors)}")
    
    lines.extend([
        "",
        f"🌡️ VIX: {vix_val:.1f} → {vix_weather}",
        f"   └ 恐怖指数。20以下は安心、30超は警戒",
        f"👥 センチメント: {sentiment} (強気{aaii_bull:.0f}%/弱気{aaii_bear:.0f}%)",
        f"   └ 個人投資家心理。逆張り指標として有効",
        f"📈 RSI: {rsi_val:.1f} → {rsi_indicator}",
        f"   └ 70超=買われすぎ、30未満=売られすぎ",
        f"🌊 CFTC: {cftc_net:,} → {pressure}",
        f"   └ 海外投機筋のポジション。マイナス=売り越し",
    ])

    if margin_balance > 0:
        margin_ratio = margin_balance / 1000000  # 百万株単位
        lines.append(f"💰 信用残: {margin_ratio:.1f}M株 (買残-売残)")
        lines.append(f"   └ 高いと将来の売り圧力、低いと買い余力")

    # Regional RSI comparison
    if regional_rsi:
        lines.extend(["", "📊 *地域別RSI*"])
        for region, rsi_value in regional_rsi.items():
            if rsi_value >= 70:
                status = "🔥"
            elif rsi_value >= 60:
                status = "🌡️"
            elif rsi_value <= 30:
                status = "❄️"
            elif rsi_value <= 40:
                status = "🌬️"
            else:
                status = "➡️"
            lines.append(f"   {region}: {rsi_value:.1f} {status}")

        # Japan comparison insight
        jp_rsi = regional_rsi.get("🇯🇵日本")
        us_rsi = regional_rsi.get("🇺🇸米国")
        world_rsi = regional_rsi.get("🌍世界")
        eu_rsi = regional_rsi.get("🇪🇺欧州")

        if jp_rsi and us_rsi and world_rsi:
            lines.append("")
            lines.append("💡 *日本 vs 世界*")

            jp_vs_us = jp_rsi - us_rsi
            jp_vs_world = jp_rsi - world_rsi

            if jp_vs_us > 10:
                lines.append(f"   🇯🇵日本は米国より過熱 (+{jp_vs_us:.0f}pt)")
            elif jp_vs_us < -10:
                lines.append(f"   🇯🇵日本は米国より出遅れ ({jp_vs_us:.0f}pt)")

            if jp_vs_world > 10:
                lines.append(f"   🇯🇵日本は世界平均より強い (+{jp_vs_world:.0f}pt)")
            elif jp_vs_world < -10:
                lines.append(f"   🇯🇵日本は世界平均より弱い ({jp_vs_world:.0f}pt)")

            if abs(jp_vs_us) <= 10 and abs(jp_vs_world) <= 10:
                lines.append("   🇯🇵日本は世界と同程度の水準")

            # Highlight strongest/weakest regions
            sorted_regions = sorted(regional_rsi.items(), key=lambda x: x[1], reverse=True)
            strongest = sorted_regions[0]
            weakest = sorted_regions[-1]
            if strongest[1] - weakest[1] > 15:
                lines.append(f"   最強: {strongest[0]} / 最弱: {weakest[0]}")

            # Japan structural analysis
            if jp_rsi and jp_rsi > 55 and (usdjpy > 140 or cftc_net < -50000):
                lines.append("")
                lines.append("⚠️ *日本市場の注意点*")

                warnings = []

                # Yen weakness analysis
                if usdjpy > 145:
                    lines.append(f"   💴 円安 {usdjpy:.0f}円 → 円建て利益膨張")
                    warnings.append("ドル建てでは米国株に劣後")

                # Foreign selling pressure
                if cftc_net < -100000:
                    lines.append(f"   🌊 外国人売り {cftc_net:,} → 海外勢は弱気")
                    warnings.append("国内勢が支えている構図")
                elif cftc_net < -50000:
                    lines.append(f"   🌊 CFTC {cftc_net:,} → 外圧あり")

                # Structural conclusion
                if usdjpy > 140 and cftc_net < -50000:
                    lines.append("   📊 結論: 「円安効果+国内フロー」による強さ")
                    lines.append("   → 外圧顕在化で調整リスクあり")

    message = "\n".join(lines)
    print(message)

    # Build image URLs
    image_urls: list[str] = []
    if chart_base_url:
        image_urls = [
            f"{chart_base_url.rstrip('/')}/charts/combined.png",
        ]

    slack_notify(slack_webhook, message, image_urls if image_urls else None)

    # Send detailed AI insights as follow-up
    insights = generate_market_insights(vix_val, rsi_val, cftc_net, aaii_bull, aaii_bear, margin_balance)
    if insights:
        slack_notify(slack_webhook, insights)


if __name__ == "__main__":
    import sys
    if "--notify-only" in sys.argv:
        notify_only()
    else:
        main()
