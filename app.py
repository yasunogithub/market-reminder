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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def slack_notify(webhook_url: str, text: str) -> None:
    if not webhook_url:
        return
    r = requests.post(webhook_url, json={"text": text}, timeout=20)
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
    return df.rename(columns={"DATE": "date", "VIXCLS": "vix"})


def fetch_cftc_finfutwk() -> pd.DataFrame:
    url = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"
    return fetch_text_as_csv(url, header=None)


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

    default_targets = "^spx,NIKKEI_OFFICIAL,fx.f,acwi.us"
    targets = [s.strip() for s in os.environ.get("RSI_TARGETS", default_targets).split(",") if s.strip()]

    # ---- VIX ----
    vix = fetch_vix_fred()
    save_csv(vix, os.path.join(DATA_DIR, "vix.csv"))
    save_line_chart(vix, "date", "vix", "VIX (FRED VIXCLS)", os.path.join(CHART_DIR, "vix.png"))
    vix_dates = pd.to_datetime(vix["date"], errors="coerce")
    vix_max = vix_dates.max()
    vix_key = str(vix_max.date()) if pd.notna(vix_max) else "unknown"
    vix_updated = changed_since_last_run("vix", vix_key)

    # ---- CFTC ----
    cftc = fetch_cftc_finfutwk()
    save_csv(cftc, os.path.join(DATA_DIR, "cftc_finfutwk_raw.csv"))
    cftc_key = str(len(cftc))
    cftc_updated = changed_since_last_run("cftc", cftc_key)

    # ---- J-Quants weekly margin ----
    margin_updated = False
    margin_note = "信用残: 未設定"
    j_token = os.environ.get("JQUANTS_TOKEN", "")
    if j_token and os.environ.get("JQUANTS_MARGIN_API_URL", ""):
        try:
            margin = fetch_jquants_weekly_margin(j_token)
            save_csv(margin, os.path.join(DATA_DIR, "margin_weekly.csv"))
            key = str(len(margin))
            for cand in ["Date", "date", "EndDate", "end_date", "TradeDate", "trade_date"]:
                if cand in margin.columns:
                    max_date = pd.to_datetime(margin[cand], errors="coerce").max()
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
        aaii, aaii_source = fetch_aaii(aaii_mode, aaii_manual_file)
        save_csv(aaii, os.path.join(DATA_DIR, "aaii.csv"))
        key = str(len(aaii))
        for cand in aaii.columns:
            if "date" in str(cand).lower():
                max_date = pd.to_datetime(aaii[cand], errors="coerce").max()
                key = str(max_date.date()) if pd.notna(max_date) else "unknown"
                break
        aaii_updated = changed_since_last_run("aaii", key)
        aaii_note = f"AAII({aaii_source}): {'更新あり' if aaii_updated else '更新なし'} ({key})"
    except Exception as e:
        aaii_note = f"AAII: 取得失敗 ({e.__class__.__name__})"

    # ---- RSI ----
    rsi_lines: list[str] = []
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

            last = px.dropna(subset=["RSI14"]).iloc[-1]
            rsi_lines.append(f"RSI {label}: {float(last['RSI14']):.1f}")
        except Exception as e:
            rsi_lines.append(f"RSI {sym}: 失敗({e.__class__.__name__})")

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
    slack_notify(slack_webhook, message)


if __name__ == "__main__":
    main()
