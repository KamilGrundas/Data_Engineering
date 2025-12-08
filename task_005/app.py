import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from outliers import smirnov_grubbs as grubbs
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas

load_dotenv()

DATA_URL = os.getenv("DATA_URL", "")


@st.cache_data
def load_data(url):
    return pd.read_csv(url)


st.sidebar.header("Data")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    df = load_data(DATA_URL)
    st.sidebar.success("Data has been refreshed.")

df = load_data(DATA_URL)

mines = df.columns.difference(
    ["Date", "Event factor", "Event probability", "Day Factor"]
).tolist()
df[mines] = df[mines].replace(",", ".", regex=True)
df[mines] = df[mines].apply(pd.to_numeric, errors="coerce")

# --- detectors: return list of (index, value) pairs ---


def iqr_outliers_list(column, multiplier=1.5):
    s = df[column]
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - multiplier * IQR
    ub = Q3 + multiplier * IQR
    mask = (s < lb) | (s > ub)
    return list(zip(s.index[mask].tolist(), s[mask].tolist()))


def z_score_outliers_list(column, threshold=3.0):
    s = df[column]
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return []
    mask = (s - m).abs() > threshold * sd
    return list(zip(s.index[mask].tolist(), s[mask].tolist()))


def ma_percent_outliers_list(column, window=5, pct_threshold=10.0):
    s = df[column]
    ma = s.rolling(window=window, min_periods=1).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (s - ma).abs() / ma.replace(0, np.nan).abs() * 100
    mask = pct > pct_threshold
    return list(zip(s.index[mask].tolist(), s[mask].tolist()))


def grubbs_outliers_list(column, alpha=0.05):
    s_nonan = df[column].dropna()
    values = s_nonan.to_numpy()
    idxs = s_nonan.index.to_numpy()
    if len(values) < 3:
        return []
    filtered = grubbs.test(values, alpha=alpha)
    # filtered is array of retained values (no outliers). Build multiset counts
    filt_list = list(filtered)
    # iterate original values and remove matched from filt_list to detect which were removed
    out = []
    filt_counts = {}
    for v in filt_list:
        filt_counts[v] = filt_counts.get(v, 0) + 1
    for i, v in enumerate(values):
        if filt_counts.get(v, 0) > 0:
            filt_counts[v] -= 1
        else:
            out.append((int(idxs[i]), float(v)))
    return out


# --- combine detectors for a mine ---
def detect_anomalies_for_mine(
    mine,
    iqr_mult,
    z_thresh,
    ma_window,
    ma_pct,
    grubbs_alpha,
    use_iqr,
    use_z,
    use_ma,
    use_grubbs,
):
    anomalies = []
    seen = set()
    if use_iqr:
        for idx, val in iqr_outliers_list(mine, multiplier=iqr_mult):
            if idx not in seen:
                anomalies.append(
                    {"index": int(idx), "value": float(val), "method": "IQR"}
                )
                seen.add(int(idx))
    if use_z:
        for idx, val in z_score_outliers_list(mine, threshold=z_thresh):
            if idx not in seen:
                anomalies.append(
                    {"index": int(idx), "value": float(val), "method": "Z-score"}
                )
                seen.add(int(idx))
    if use_ma:
        for idx, val in ma_percent_outliers_list(
            mine, window=ma_window, pct_threshold=ma_pct
        ):
            if idx not in seen:
                anomalies.append(
                    {"index": int(idx), "value": float(val), "method": "MA %"}
                )
                seen.add(int(idx))
    if use_grubbs:
        for idx, val in grubbs_outliers_list(mine, alpha=grubbs_alpha):
            if idx not in seen:
                anomalies.append(
                    {"index": int(idx), "value": float(val), "method": "Grubbs"}
                )
                seen.add(int(idx))
    anomalies.sort(key=lambda x: x["index"])
    return anomalies


# --- UI controls ---
st.title("Mining Data Visualization and Outlier Detection")

st.sidebar.header("Plot Settings")
chart_type = st.sidebar.selectbox(
    "Chart Type", options=["line", "bar", "stacked"], index=0
)
if chart_type == "stacked":
    selected_mines = st.sidebar.multiselect(
        "Select Mines (stacked)", options=mines, default=mines[:3]
    )
    if not selected_mines:
        st.warning("Select at least one mine for stacked chart.")
        st.stop()
else:
    selected_mine = st.sidebar.selectbox("Select Mine", options=mines, index=0)

show_trend = st.sidebar.checkbox("Show Trend", value=True)
trend_degree = st.sidebar.selectbox(
    "Trend Polynomial Degree", options=[1, 2, 3, 4], index=0
)

st.sidebar.markdown("Detectors (main param each)")
iqr_mult = st.sidebar.slider("IQR multiplier", 0.1, 10.0, 1.5, 0.1)
z_thresh = st.sidebar.slider("Z-score threshold", 0.5, 10.0, 3.0, 0.1)
ma_window = st.sidebar.slider("MA window size", 1, 365, 5, 1)
ma_pct = st.sidebar.slider("MA percent threshold", 0.1, 1000.0, 10.0, 0.1)
grubbs_alpha = st.sidebar.slider("Grubbs alpha", 0.0001, 0.5, 0.05, 0.0001)

st.sidebar.markdown("Include detectors")
use_iqr = st.sidebar.checkbox("Include IQR", value=True)
use_z = st.sidebar.checkbox("Include Z-score", value=True)
use_ma = st.sidebar.checkbox("Include Moving Average", value=False)
use_grubbs = st.sidebar.checkbox("Include Grubbs", value=False)

exclude_outliers_from_stats = st.sidebar.checkbox(
    "Exclude outliers from stats", value=False
)

# --- prepare x axis ---
plot_df = df.copy()
if "Date" in plot_df.columns:
    x = pd.to_datetime(plot_df["Date"])
else:
    x = plot_df.index

# --- plotting and UI stats ---
fig = go.Figure()

if chart_type in ["line", "bar"]:
    y = plot_df[selected_mine]

    anomalies = detect_anomalies_for_mine(
        selected_mine,
        iqr_mult,
        z_thresh,
        ma_window,
        ma_pct,
        grubbs_alpha,
        use_iqr,
        use_z,
        use_ma,
        use_grubbs,
    )
    out_idx = [a["index"] for a in anomalies]
    mask = y.index.isin(out_idx)

    if exclude_outliers_from_stats:
        y_stats = y[~mask]
    else:
        y_stats = y

    if chart_type == "line":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=selected_mine))
    else:
        fig.add_trace(go.Bar(x=x, y=y, name=selected_mine))

    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker=dict(size=10, symbol="x", color="red"),
                name="Outliers",
            )
        )

    if show_trend:
        xn = np.arange(len(y))
        valid = ~np.isnan(y)
        if valid.sum() > trend_degree:
            coeffs = np.polyfit(xn[valid], y[valid], deg=trend_degree)
            trend = np.polyval(coeffs, xn)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=trend,
                    mode="lines",
                    name=f"Trend deg {trend_degree}",
                    line=dict(dash="dash"),
                )
            )

    fig.update_layout(
        title=f"{selected_mine}",
        xaxis_title="Date" if "Date" in plot_df.columns else "Index",
        yaxis_title="Value",
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    stats = {
        "mean": float(y_stats.mean()) if len(y_stats.dropna()) > 0 else None,
        "std dev": float(y_stats.std()) if len(y_stats.dropna()) > 0 else None,
        "median": float(y_stats.median()) if len(y_stats.dropna()) > 0 else None,
        "IQR": float(y_stats.quantile(0.75) - y_stats.quantile(0.25))
        if len(y_stats.dropna()) > 0
        else None,
        "detected_outliers": int(mask.sum()),
    }
    st.subheader("Basic statistics")
    st.table(pd.DataFrame({selected_mine: stats}).T)

else:
    selected = selected_mines
    y_vals = plot_df[selected].fillna(0)
    for m in selected:
        fig.add_trace(go.Bar(x=x, y=y_vals[m], name=m))

    fig.update_layout(
        barmode="relative",
        title=f"Stacked: {', '.join(selected)}",
        xaxis_title="Date" if "Date" in plot_df.columns else "Index",
        yaxis_title="Value",
        legend=dict(orientation="h"),
    )

    # draw per-mine outlier markers using indices from detect_anomalies_for_mine
    for mine in selected:
        series = plot_df[mine]
        anomalies = detect_anomalies_for_mine(
            mine,
            iqr_mult,
            z_thresh,
            ma_window,
            ma_pct,
            grubbs_alpha,
            use_iqr,
            use_z,
            use_ma,
            use_grubbs,
        )
        out_idx = [a["index"] for a in anomalies]
        mask = series.index.isin(out_idx)
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=series[mask],
                    mode="markers",
                    marker=dict(size=9, symbol="x", color="red"),
                    name=f"{mine} outliers",
                )
            )

    # aggregate trendline
    if show_trend:
        agg = plot_df[selected].sum(axis=1)
        xn = np.arange(len(agg))
        valid = ~np.isnan(agg)
        if valid.sum() > trend_degree:
            coeffs = np.polyfit(xn[valid], agg[valid], deg=trend_degree)
            trend = np.polyval(coeffs, xn)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=trend,
                    mode="lines",
                    name=f"Aggregate trend deg {trend_degree}",
                    line=dict(dash="dash"),
                )
            )

    st.plotly_chart(fig, use_container_width=True)

    # compute per-mine stats table using anomaly indices
    stats_rows = {}
    for mine in selected:
        s = plot_df[mine]
        anomalies = detect_anomalies_for_mine(
            mine,
            iqr_mult,
            z_thresh,
            ma_window,
            ma_pct,
            grubbs_alpha,
            use_iqr,
            use_z,
            use_ma,
            use_grubbs,
        )
        out_idx = [a["index"] for a in anomalies]
        mask = s.index.isin(out_idx)
        if exclude_outliers_from_stats:
            s_use = s[~mask]
        else:
            s_use = s
        stats_rows[mine] = {
            "mean": float(s_use.mean()) if len(s_use.dropna()) > 0 else None,
            "std dev": float(s_use.std()) if len(s_use.dropna()) > 0 else None,
            "median": float(s_use.median()) if len(s_use.dropna()) > 0 else None,
            "IQR": float(s_use.quantile(0.75) - s_use.quantile(0.25))
            if len(s_use.dropna()) > 0
            else None,
            "detected_outliers_count": int(len(out_idx)),
        }

    st.markdown("---")
    st.subheader("Mines statistics")
    st.table(pd.DataFrame(stats_rows).T)

# ---------------- PDF generation helpers ----------------


def make_matplotlib_png_for_series(
    x_vals, series, anomalies_mask, title, add_trend=True, degree=1
):
    figm, ax = plt.subplots(figsize=(8, 4.5))
    if len(x_vals) > 0 and isinstance(x_vals[0], (np.datetime64, pd.Timestamp)):
        ax.plot(x_vals, series, marker="o", linestyle="-")
        ax.plot(x_vals[anomalies_mask], series[anomalies_mask], "rx", label="Outliers")
        ax.set_xlabel("Date")
    else:
        ax.plot(series.values, marker="o", linestyle="-")
        ax.plot(
            np.where(anomalies_mask)[0], series[anomalies_mask], "rx", label="Outliers"
        )
        ax.set_xlabel("Index")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if add_trend:
        xn = np.arange(len(series))
        valid = ~np.isnan(series)
        if valid.sum() > degree:
            coeffs = np.polyfit(xn[valid], series[valid], deg=degree)
            trend = np.polyval(coeffs, xn)
            if isinstance(x_vals[0], (np.datetime64, pd.Timestamp)):
                ax.plot(x_vals, trend, linestyle="--", label=f"Trend deg {degree}")
            else:
                ax.plot(trend, linestyle="--", label=f"Trend deg {degree}")
    ax.legend(loc="upper right")
    figm.tight_layout()
    buf = BytesIO()
    figm.savefig(buf, format="png", dpi=150)
    plt.close(figm)
    buf.seek(0)
    return buf


def build_pdf_report(selected_list, include_aggregate, params, detectors, degree):
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    w, h = A4
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Title page
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, h - 60, "Mining Data - Detailed Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 80, f"Generated: {gen_time}")
    c.drawString(40, h - 95, f"Included mines: {', '.join(selected_list)}")
    if include_aggregate:
        c.drawString(40, h - 110, "Including aggregate (sum) of selected mines.")
    c.showPage()

    # Overview table
    table_data = [["Mine", "Mean", "Std dev", "Median", "IQR", "Detected anomalies"]]
    for m in selected_list:
        s = df[m]
        anomalies = detect_anomalies_for_mine(
            m,
            params["iqr_mult"],
            params["z_thresh"],
            params["ma_window"],
            params["ma_pct"],
            params["grubbs_alpha"],
            detectors["iqr"],
            detectors["z"],
            detectors["ma"],
            detectors["grubbs"],
        )
        mask_idx = s.index.isin([a["index"] for a in anomalies])
        if exclude_outliers_from_stats:
            s_stats = s[~mask_idx]
        else:
            s_stats = s
        mean_v = f"{float(s_stats.mean()):.3f}" if len(s_stats.dropna()) > 0 else "N/A"
        std_v = f"{float(s_stats.std()):.3f}" if len(s_stats.dropna()) > 0 else "N/A"
        med_v = f"{float(s_stats.median()):.3f}" if len(s_stats.dropna()) > 0 else "N/A"
        iqr_v = (
            f"{float(s_stats.quantile(0.75) - s_stats.quantile(0.25)):.3f}"
            if len(s_stats.dropna()) > 0
            else "N/A"
        )
        table_data.append([m, mean_v, std_v, med_v, iqr_v, str(len(anomalies))])

    table = Table(table_data, colWidths=[120, 70, 70, 70, 70, 70])
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FAFAFA")),
        ]
    )
    table.setStyle(style)
    tw, th = table.wrapOn(c, w - 80, h - 200)
    table.drawOn(c, 40, h - 120 - th)
    c.showPage()

    # per-mine pages
    for m in selected_list:
        s = df[m]
        anomalies = detect_anomalies_for_mine(
            m,
            params["iqr_mult"],
            params["z_thresh"],
            params["ma_window"],
            params["ma_pct"],
            params["grubbs_alpha"],
            detectors["iqr"],
            detectors["z"],
            detectors["ma"],
            detectors["grubbs"],
        )
        mask_idx = s.index.isin([a["index"] for a in anomalies])
        png_buf = make_matplotlib_png_for_series(
            x,
            s,
            mask_idx.values if hasattr(mask_idx, "values") else mask_idx,
            title=m,
            add_trend=True,
            degree=degree,
        )

        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, h - 60, f"Mine: {m}")
        c.setFont("Helvetica", 10)
        c.drawRightString(w - 40, h - 60, f"Generated: {gen_time}")
        img = ImageReader(png_buf)
        img_w = w - 80
        img_h = img_w * 0.45
        c.drawImage(
            img,
            40,
            h - 90 - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        stats_for_m = {
            "Mean": (
                float(s[~mask_idx].mean())
                if exclude_outliers_from_stats and len(s[~mask_idx].dropna()) > 0
                else float(s.mean())
                if len(s.dropna()) > 0
                else None
            ),
            "Std dev": (
                float(s[~mask_idx].std())
                if exclude_outliers_from_stats and len(s[~mask_idx].dropna()) > 0
                else float(s.std())
                if len(s.dropna()) > 0
                else None
            ),
            "Median": (
                float(s[~mask_idx].median())
                if exclude_outliers_from_stats and len(s[~mask_idx].dropna()) > 0
                else float(s.median())
                if len(s.dropna()) > 0
                else None
            ),
            "IQR": (
                float(s[~mask_idx].quantile(0.75) - s[~mask_idx].quantile(0.25))
                if exclude_outliers_from_stats and len(s[~mask_idx].dropna()) > 0
                else float(s.quantile(0.75) - s.quantile(0.25))
                if len(s.dropna()) > 0
                else None
            ),
            "Detected anomalies": len(anomalies),
        }

        trows = [["Metric", "Value"]]
        for k, v in stats_for_m.items():
            trows.append(
                [
                    k,
                    f"{v:.3f}"
                    if isinstance(v, float)
                    else (str(v) if v is not None else "N/A"),
                ]
            )
        t2 = Table(trows, colWidths=[200, 200])
        t2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFFFF")),
                ]
            )
        )
        tw2, th2 = t2.wrapOn(c, w - 80, h - 140 - img_h)
        t2.drawOn(c, 40, h - 110 - img_h - th2)

        y = h - 130 - img_h - th2
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Detected anomalies (by detector):")
        y -= 18
        c.setFont("Helvetica", 10)
        if anomalies:
            for an in anomalies:
                idx = an["index"]
                val = an["value"]
                ma_val = (
                    df[m]
                    .rolling(window=params["ma_window"], min_periods=1)
                    .mean()
                    .iloc[idx]
                    if params["ma_window"] > 0
                    else np.nan
                )
                kind = "spike" if val > ma_val else "drop"
                line = (
                    f"- index {idx}: value={val:.3f} detector={an['method']} ({kind})"
                )
                c.drawString(48, y, line)
                y -= 12
                if y < 80:
                    c.showPage()
                    y = h - 40
                    c.setFont("Helvetica", 10)
        else:
            c.drawString(48, y, "No anomalies detected with chosen detectors/params.")
            y -= 12

        c.showPage()

    # aggregate
    if include_aggregate:
        agg = plot_df[selected_list].sum(axis=1)
        # detectors on agg
        Q1 = agg.quantile(0.25)
        Q3 = agg.quantile(0.75)
        IQR = Q3 - Q1
        mask_i = (
            (agg < Q1 - params["iqr_mult"] * IQR)
            | (agg > Q3 + params["iqr_mult"] * IQR)
            if detectors["iqr"]
            else pd.Series(False, index=agg.index)
        )
        if detectors["z"]:
            mean_agg = agg.mean()
            std_agg = agg.std()
            mask_z = (agg - mean_agg).abs() > params["z_thresh"] * std_agg
        else:
            mask_z = pd.Series(False, index=agg.index)
        if detectors["ma"]:
            ma = agg.rolling(window=params["ma_window"], min_periods=1).mean()
            pct = (agg - ma).abs() / ma.replace(0, np.nan).abs() * 100
            mask_ma = pct > params["ma_pct"]
        else:
            mask_ma = pd.Series(False, index=agg.index)
        if detectors["grubbs"]:
            if len(agg.dropna()) >= 3:
                filtered = grubbs.test(
                    agg.dropna().to_numpy(), alpha=params["grubbs_alpha"]
                )
                filtered_list = list(filtered)
                mask_g = agg.isin([v for v in agg if v not in filtered_list])
            else:
                mask_g = pd.Series(False, index=agg.index)
        else:
            mask_g = pd.Series(False, index=agg.index)
        agg_mask = mask_i | mask_z | mask_ma | mask_g

        png_buf = make_matplotlib_png_for_series(
            x,
            agg,
            agg_mask.values if hasattr(agg_mask, "values") else agg_mask,
            title="Aggregate (sum)",
            add_trend=True,
            degree=degree,
        )
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, h - 60, "Aggregate (sum) of selected mines")
        img = ImageReader(png_buf)
        img_w = w - 80
        img_h = img_w * 0.45
        c.drawImage(
            img,
            40,
            h - 90 - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        s = agg
        stats_for_agg = {
            "Mean": float(s[~agg_mask].mean())
            if exclude_outliers_from_stats and len(s[~agg_mask].dropna()) > 0
            else float(s.mean())
            if len(s.dropna()) > 0
            else None,
            "Std dev": float(s[~agg_mask].std())
            if exclude_outliers_from_stats and len(s[~agg_mask].dropna()) > 0
            else float(s.std())
            if len(s.dropna()) > 0
            else None,
            "Median": float(s[~agg_mask].median())
            if exclude_outliers_from_stats and len(s[~agg_mask].dropna()) > 0
            else float(s.median())
            if len(s.dropna()) > 0
            else None,
            "IQR": float(s.quantile(0.75) - s.quantile(0.25))
            if len(s.dropna()) > 0
            else None,
            "Detected anomalies": int(agg_mask.sum()),
        }
        trows = [["Metric", "Value"]]
        for k, v in stats_for_agg.items():
            trows.append(
                [
                    k,
                    f"{v:.3f}"
                    if isinstance(v, float)
                    else (str(v) if v is not None else "N/A"),
                ]
            )
        t_agg = Table(trows, colWidths=[200, 200])
        t_agg.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFFFF")),
                ]
            )
        )
        tw3, th3 = t_agg.wrapOn(c, w - 80, h - 120 - img_h)
        t_agg.drawOn(c, 40, h - 110 - img_h - th3)
        c.showPage()

    c.save()
    pdf_buf.seek(0)
    return pdf_buf


# prepare params/detectors
params = {
    "iqr_mult": iqr_mult,
    "z_thresh": z_thresh,
    "ma_window": ma_window,
    "ma_pct": ma_pct,
    "grubbs_alpha": grubbs_alpha,
}
detectors = {"iqr": use_iqr, "z": use_z, "ma": use_ma, "grubbs": use_grubbs}
degree = trend_degree

# single download button (generates and downloads)
if chart_type in ["line", "bar"]:
    sel = [selected_mine]
else:
    sel = selected_mines

if st.button("Download PDF report"):
    pdf_bytes = build_pdf_report(sel, True, params, detectors, degree)
    st.download_button(
        "Click to download PDF",
        data=pdf_bytes.getvalue(),
        file_name=f"mining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )
