"""
Patch-ready KPI chart helpers for HVAC Research Modeling Suite v3

Adds:
- KPI charts for Energy Consumption
- KPI charts for Comfort Deviation
- KPI charts for Mean Degradation Index
- KPI charts for Carbon Emissions
- kpi_summary.csv export
- saved PNG figures inside figures/

Use:
1) Put this file next to your Streamlit app.
2) Import:
   from kpi_chart_patch_v3 import render_kpi_outputs_from_result, render_kpi_outputs_from_folder
3) After a model run:
   render_kpi_outputs_from_result(result)
4) In Exports and Results tab:
   render_kpi_outputs_from_folder(p)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def build_kpi_summary_from_summary_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "scenario_combo_3axis": r.get("scenario_combo_3axis", ""),
            "strategy": r.get("strategy", ""),
            "severity": r.get("severity", ""),
            "climate": r.get("climate", ""),
            "Energy Consumption": r.get("Total Energy MWh", None),
            "Comfort Deviation": r.get("Mean Comfort Deviation C", None),
            "Mean Degradation Index": r.get("Mean Degradation Index", None),
            "Carbon Emissions": r.get("Total CO2 tonne", None),
        })
    return pd.DataFrame(rows)


def _make_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df[x_col].astype(str), pd.to_numeric(df[y_col], errors="coerce"), marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def _save_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path):
    fig = _make_line_chart(df, x_col, y_col, title)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_kpi_figures_from_summary(summary_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    kpi_df = build_kpi_summary_from_summary_csv(summary_df)
    kpi_csv = output_dir / "kpi_summary.csv"
    kpi_df.to_csv(kpi_csv, index=False)

    x_col = "scenario_combo_3axis"
    files = {}
    for y_col, filename in [
        ("Energy Consumption", "kpi_energy.png"),
        ("Comfort Deviation", "kpi_comfort.png"),
        ("Mean Degradation Index", "kpi_degradation.png"),
        ("Carbon Emissions", "kpi_carbon.png"),
    ]:
        out_png = figures_dir / filename
        _save_line_chart(kpi_df, x_col, y_col, y_col, out_png)
        files[y_col] = str(out_png)

    return {"kpi_csv": str(kpi_csv), **files}


def render_kpi_outputs_from_result(result: dict):
    summary_path = Path(result["summary_csv"])
    output_dir = summary_path.parent
    saved = save_kpi_figures_from_summary(summary_path, output_dir)

    st.markdown("### KPI charts")
    kpi_df = pd.read_csv(saved["kpi_csv"])
    st.dataframe(kpi_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(saved["Energy Consumption"], caption="Energy Consumption")
        st.image(saved["Mean Degradation Index"], caption="Mean Degradation Index")
    with c2:
        st.image(saved["Comfort Deviation"], caption="Comfort Deviation")
        st.image(saved["Carbon Emissions"], caption="Carbon Emissions")

    with open(saved["kpi_csv"], "rb") as f:
        st.download_button("Download KPI summary CSV", f.read(), file_name="kpi_summary.csv", mime="text/csv")


def render_kpi_outputs_from_folder(folder: str | Path):
    folder = Path(folder)
    summary_candidates = [
        folder / "matrix_summary.csv",
        folder / "one_axis_severity_summary.csv",
        folder / "one_axis_strategy_summary.csv",
        folder / "three_axis_summary.csv",
    ]
    summary_path = None
    for p in summary_candidates:
        if p.exists():
            summary_path = p
            break

    if summary_path is None:
        st.info("No summary CSV found yet for KPI charts.")
        return

    saved = save_kpi_figures_from_summary(summary_path, folder)
    kpi_df = pd.read_csv(saved["kpi_csv"])

    st.markdown("### KPI summary")
    st.dataframe(kpi_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(saved["Energy Consumption"], caption="Energy Consumption")
        st.image(saved["Mean Degradation Index"], caption="Mean Degradation Index")
    with c2:
        st.image(saved["Comfort Deviation"], caption="Comfort Deviation")
        st.image(saved["Carbon Emissions"], caption="Carbon Emissions")


INTEGRATION_NOTES = """
Add this import near the top of your Streamlit app:
from kpi_chart_patch_v3 import render_kpi_outputs_from_result, render_kpi_outputs_from_folder

In your Run selected model block, after showing the summary dataframe, add:
render_kpi_outputs_from_result(result)

In your Exports and Results tab, after the CSV previews, add:
render_kpi_outputs_from_folder(p)
"""
