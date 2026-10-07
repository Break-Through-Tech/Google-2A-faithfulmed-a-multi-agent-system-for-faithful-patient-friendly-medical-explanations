from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Baseline Evaluation Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_results() -> pd.DataFrame:
    frame = pd.read_csv("baseline_evaluation_results.csv")
    return frame.rename(
        columns={
            "sample_name": "Sample name",
            "original_transcription": "Transcription",
            "simplified_output": "Simplified transcription",
            "flesch_kincaid": "Flesch-Kincaid grade",
            "smog": "SMOG index",
            "jargon_density": "Jargon density",
            "word_count": "Word count",
        }
    )


def histogram(frame: pd.DataFrame, metric: str, bins: int = 8) -> tuple[go.Figure, list[str]]:
    values = frame[metric].to_numpy() if not frame.empty else np.array([0.0])
    value_min, value_max = values.min(), values.max()
    if value_min == value_max:
        value_min -= 0.5
        value_max += 0.5
    edges = np.linspace(value_min, value_max, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    labels = [f"{edges[index]:.2f}–{edges[index + 1]:.2f}" for index in range(bins)]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            customdata=np.arange(bins),
            marker_color="#2f6f8f",
            hovertemplate="Range: %{x}<br>Outputs: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=12, b=48),
        xaxis_title=None,
        yaxis_title="Outputs",
        showlegend=False,
        clickmode="event+select",
    )
    return figure, labels


def selected_rows(frame: pd.DataFrame, metric: str, selected_bins: list[int], bins: int = 8) -> pd.DataFrame:
    if not selected_bins or frame.empty:
        return frame
    values = frame[metric].to_numpy()
    value_min, value_max = values.min(), values.max()
    if value_min == value_max:
        value_min -= 0.5
        value_max += 0.5
    edges = np.linspace(value_min, value_max, bins + 1)
    bin_indexes = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    return frame[np.isin(bin_indexes, selected_bins)]


st.title("Baseline Evaluation Results")
st.caption("Select histogram bars to explore the evaluation results, then open any sample card for the full text comparison.")

data = load_results()
baseline = {
    "num_outputs": len(data),
    "num_valid_outputs": data["Simplified transcription"].fillna("").astype(str).str.strip().ne("").sum(),
    "num_refusals": int(data["refusal"].sum()),
    "refusal_rate": data["refusal"].mean(),
    "avg_flesch_kincaid": data["Flesch-Kincaid grade"].mean(),
    "percent_at_or_below_grade_8": data["meets_grade_8"].mean() * 100,
    "avg_smog": data["SMOG index"].mean(),
    "avg_jargon_density": data["Jargon density"].mean(),
    "avg_word_count": data["Word count"].mean(),
}
summary = st.columns(5)
summary[0].metric("Outputs", baseline["num_outputs"])
summary[1].metric("Valid outputs", baseline["num_valid_outputs"])
summary[2].metric("Refusals", baseline["num_refusals"])
summary[3].metric("Refusal rate", f"{baseline['refusal_rate']:.1%}")
summary[4].metric("At or below grade 8", f"{baseline['percent_at_or_below_grade_8']:.0f}%")

st.subheader("Metric distributions")
metric_columns = ["Flesch-Kincaid grade", "SMOG index", "Jargon density", "Word count"]
average_labels = {
    "Flesch-Kincaid grade": baseline["avg_flesch_kincaid"],
    "SMOG index": baseline["avg_smog"],
    "Jargon density": baseline["avg_jargon_density"],
    "Word count": baseline["avg_word_count"],
}
selected_filters: dict[str, list[int]] = {}
working_data = data

for row_start in range(0, len(metric_columns), 2):
    columns = st.columns(2)
    for column, metric in zip(columns, metric_columns[row_start : row_start + 2]):
        figure, labels = histogram(working_data, metric)
        with column:
            st.markdown(f"**{metric}**")
            event = st.plotly_chart(
                figure,
                width="stretch",
                key=f"histogram-{metric}",
                on_select="rerun",
                selection_mode="points",
            )
            points = event.selection.points if event and event.selection else []
            selected_filters[metric] = [point["point_index"] for point in points]
            st.caption(f"Baseline average: {average_labels[metric]:.2f}")
        working_data = selected_rows(working_data, metric, selected_filters[metric])

active_metric = next((metric for metric, filters in selected_filters.items() if filters), None)
filtered_data = working_data

@st.dialog("Sample details", width="large")
def show_sample_details(sample: dict[str, object]) -> None:
    st.subheader(str(sample["Sample name"]))
    details = st.columns(4)
    for column, metric in zip(details, metric_columns):
        with column:
            st.caption(metric)
            st.write(f"{float(sample[metric]):.2f}")

    transcription, simplified = st.columns(2)
    with transcription:
        st.markdown("**Transcription**")
        st.write(sample["Transcription"])
    with simplified:
        st.markdown("**Simplified transcription**")
        st.write(sample["Simplified transcription"])


st.subheader("Filtered samples")
if active_metric:
    st.info(f"Showing {len(filtered_data)} outputs in the selected {active_metric} bin.")
else:
    st.info("Select a bar in any histogram to filter these samples.")


def preview(value: object, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."

if filtered_data.empty:
    st.warning("No samples match the selected bins.")
else:
    cards = st.columns(2)
    for index, (_, sample) in enumerate(filtered_data.iterrows()):
        with cards[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{sample['Sample name']}**")
                metric_values = st.columns(4)
                for column, metric in zip(metric_values, metric_columns):
                    with column:
                        st.caption(metric)
                        st.write(f"{sample[metric]:.2f}")
                st.caption(f"Input: {preview(sample['Transcription'])}")
                st.caption(f"Simplified: {preview(sample['Simplified transcription'])}")
                if st.button("View details", key=f"details-{sample['Sample name']}", width="stretch"):
                    show_sample_details(sample.to_dict())
