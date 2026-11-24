import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ohe_engine import (
    OUTPUT_DIR,
    OHEInspectionEngine,
    generate_synthetic_calibration_csv,
    generate_synthetic_main_csv,
)

st.set_page_config(page_title="25kV AC OHE Automated Inspection", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(path: str, max_display_rows: int = 25000) -> pd.DataFrame:
    """Loads processed or raw OHE CSV with smart stride downsampling for smooth rendering."""
    if not os.path.exists(path):
        return pd.DataFrame()
    total_lines = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
    stride = max(1, total_lines // max_display_rows)

    if stride > 1:
        rows = []
        for idx, chunk in enumerate(pd.read_csv(path, chunksize=50000)):
            sampled = chunk.iloc[::stride].copy()
            rows.append(sampled)
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.read_csv(path)

    if "Corrected_Height_mm" not in df.columns:
        engine = OHEInspectionEngine()
        cal_path = os.path.join(OUTPUT_DIR, "calibration_500m_to_and_fro.csv")
        if not os.path.exists(cal_path):
            generate_synthetic_calibration_csv(cal_path)
        engine.calibrate_to_and_fro(cal_path)
        df, _ = engine.process_dataframe(df)

    return df


def main() -> None:
    st.title("Indian Railways 25kV AC OHE Automated Inspection Dashboard")
    st.caption("Roof-Mounted Locomotive Camera | Cant-Decoupled Physics + SciML + Regula Falsi + RDSO Compliance")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    default_processed = os.path.join(OUTPUT_DIR, "processed_ohe_results.csv")
    default_synthetic = os.path.join(OUTPUT_DIR, "synthetic_ohe_600k.csv")

    with st.sidebar:
        st.header("Data & Playback Controls")
        custom_path = st.text_input("CSV Path", value=default_processed if os.path.exists(default_processed) else "")
        if st.button("Generate & Load 600k Synthetic Dataset"):
            with st.spinner("Generating 600,000-row dataset..."):
                gen_path = generate_synthetic_main_csv(default_synthetic, n_rows=600000)
                cal_path = generate_synthetic_calibration_csv(os.path.join(OUTPUT_DIR, "calibration_500m_to_and_fro.csv"))
                eng = OHEInspectionEngine()
                eng.calibrate_to_and_fro(cal_path)
                eng.run_bulk_chunked(gen_path, chunksize=60000)
            st.cache_data.clear()
            st.rerun()

        active_path = custom_path.strip() if custom_path.strip() else default_synthetic
        if not os.path.exists(active_path):
            active_path = generate_synthetic_main_csv(default_synthetic, n_rows=600000)

        max_pts = st.slider("Max Downsampled Plot Points", min_value=1000, max_value=30000, value=12000, step=1000)
        window_span = st.slider("Window Span (points)", min_value=100, max_value=2000, value=500, step=50)

    df = load_dataset(active_path, max_display_rows=max_pts)
    if df.empty:
        st.error("No data available to display.")
        return

    cursor = st.slider("Playback / Inspection Cursor Index", min_value=0, max_value=max(0, len(df) - 1), value=0)
    start_idx = max(0, cursor - window_span // 2)
    end_idx = min(len(df), start_idx + window_span)
    win = df.iloc[start_idx:end_idx]
    curr = df.iloc[cursor]

    # Live Telemetry Header
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mast Number", str(curr.get("Mast_No", "N/A")))
    c2.metric("KM Post", str(curr.get("KM_Post", "N/A")))
    c3.metric("Cumulative Distance (Col C)", f"{float(curr.get('CumDist_m', 0.0)):.1f} m")
    c4.metric("Track State", f"{curr.get('Track_State', 'STRAIGHT')} ({float(curr.get('Cant_mm', 0.0)):.1f} mm cant)")
    anomaly_txt = str(curr.get("Anomaly_Detail", "OK"))
    c5.metric("RDSO Status", "ALERT" if bool(curr.get("RDSO_Anomaly", False)) else "NOMINAL", delta=anomaly_txt)

    if bool(curr.get("RDSO_Anomaly", False)):
        st.error(f"RDSO Violation at Chainage {float(curr['CumDist_m']):.1f} m: {anomaly_txt}")

    # View 1: Top View (Plan View)
    st.subheader("View 1 — Top View (Plan View: Stagger, Adjacent Wire & Takeover Point)")
    fig_top = go.Figure()
    fig_top.add_trace(go.Scatter(
        x=win["CumDist_m"], y=np.zeros(len(win)),
        mode="lines", name="Track Centerline (0 mm)", line=dict(color="#888888", dash="dash")
    ))
    fig_top.add_trace(go.Scatter(
        x=win["CumDist_m"], y=win["Corrected_Stagger_mm"],
        mode="lines", name="Primary Wire Stagger (mm)", line=dict(color="#00B4D8", width=2.5)
    ))
    fig_top.add_trace(go.Scatter(
        x=win["CumDist_m"], y=win["Next_Wire_Stagger_mm"],
        mode="lines", name="Next-to-be-Primary Wire (mm)", line=dict(color="#F77F00", width=2)
    ))
    takeovers = win[win["Takeover_Point"] == True]
    if not takeovers.empty:
        fig_top.add_trace(go.Scatter(
            x=takeovers["CumDist_m"], y=takeovers["Corrected_Stagger_mm"],
            mode="markers", name="Takeover / Handover Point",
            marker=dict(color="#E63946", size=10, symbol="diamond")
        ))
    fig_top.update_layout(xaxis_title="Cumulative Distance (m)", yaxis_title="Lateral Stagger (mm)", height=320)
    st.plotly_chart(fig_top, use_container_width=True)

    # View 2: Side View (Elevation Profile)
    st.subheader("View 2 — Side View (Contact Wire Height, Gradient & RDSO Envelope)")
    fig_side = go.Figure()
    fig_side.add_trace(go.Scatter(
        x=win["CumDist_m"], y=win["Corrected_Height_mm"],
        mode="lines", name="Contact Wire Height (mm)", line=dict(color="#2A9D8F", width=2.5)
    ))
    fig_side.add_hline(y=5800, line_dash="dot", line_color="orange", annotation_text="RDSO Upper Normal (5800 mm)")
    fig_side.add_hline(y=5500, line_dash="dot", line_color="orange", annotation_text="RDSO Lower Normal (5500 mm)")
    fig_side.add_hline(y=4800, line_dash="dash", line_color="red", annotation_text="RDSO Min Flag (4800 mm)")
    fig_side.add_hline(y=6000, line_dash="dash", line_color="red", annotation_text="RDSO Max Flag (6000 mm)")
    fig_side.update_layout(xaxis_title="Cumulative Distance (m)", yaxis_title="Height Above Rail (mm)", height=320)
    st.plotly_chart(fig_side, use_container_width=True)

    # View 3: Loco Pilot's View (Cab/Roof Perspective with Camera Roll Tilt)
    st.subheader("View 3 — Loco Pilot's View (Camera-Tilt-Compensated Forward Perspective)")
    roll_rad = np.radians(float(curr.get("Decoupled_Roll_deg", 0.0)))
    p_x = float(curr.get("Corrected_Stagger_mm", 0.0))
    p_y = float(curr.get("Corrected_Height_mm", 5600.0))
    n_x = float(curr.get("Next_Wire_Stagger_mm", -300.0))
    n_y = p_y + 120.0

    # Rotate rail & wire points by camera roll angle to depict roof camera perspective
    def rot(x_val: float, y_val: float) -> tuple:
        xr = x_val * np.cos(roll_rad) - (y_val - 5600.0) * np.sin(roll_rad)
        yr = 5600.0 + x_val * np.sin(roll_rad) + (y_val - 5600.0) * np.cos(roll_rad)
        return float(xr), float(yr)

    rail_l_x, rail_l_y = rot(-838.0, 4600.0)
    rail_r_x, rail_r_y = rot(838.0, 4600.0)
    pw_x, pw_y = rot(p_x, p_y)
    nw_x, nw_y = rot(n_x, n_y)

    fig_cab = go.Figure()
    fig_cab.add_trace(go.Scatter(
        x=[rail_l_x, rail_r_x], y=[rail_l_y, rail_r_y],
        mode="lines+markers", name="Tilted Rail Gauge (1676 mm)", line=dict(color="#6C757D", width=4)
    ))
    fig_cab.add_trace(go.Scatter(
        x=[pw_x], y=[pw_y],
        mode="markers+text", name="Primary Contact Wire",
        marker=dict(color="#0077B6", size=14), text=["PRIMARY"], textposition="top center"
    ))
    fig_cab.add_trace(go.Scatter(
        x=[nw_x], y=[nw_y],
        mode="markers+text", name="Next-to-be-Primary Wire",
        marker=dict(color="#E76F51", size=14), text=["NEXT WIRE"], textposition="top center"
    ))
    fig_cab.add_trace(go.Scatter(
        x=[pw_x, nw_x], y=[pw_y, nw_y],
        mode="lines", name=f"Wire Separation ({float(curr.get('Wire_Gap_mm', 0.0)):.1f} mm)",
        line=dict(color="#E63946", dash="dash", width=2)
    ))
    fig_cab.update_layout(
        xaxis=dict(title="Lateral Frame Axis (mm)", range=[-1200, 1200]),
        yaxis=dict(title="Vertical Elevation Axis (mm)", range=[4400, 6200]),
        height=360,
    )
    st.plotly_chart(fig_cab, use_container_width=True)

    # Metrics & Report Summary if available
    report_path = os.path.join(OUTPUT_DIR, "inspection_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            rep = json.load(f)
        with st.expander("View Exported Error Metrics, Confusion Matrices & Interpolated GPS Gaps"):
            st.json(rep)


if __name__ == "__main__":
    main()