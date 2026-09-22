"""
visuals.py
----------
Interactive Plotly visualizations for the MindType editorial research platform:
  1. Full-Width Keystroke Rhythm Chart (keystroke sequence vs interval, baseline band, hesitation spikes)
  2. Baseline vs. Session Diverging Chart (relative shifts where 100 = personal baseline)
  3. Typing Speed & Cadence Comparison (WPM and average inter-key ms)
  4. Hesitation & Pause Distribution Chart (duration breakdown <500ms, 500-1000ms, 1-2s, >2s)
  5. Correction Activity Chart (baseline vs assessment backspaces & burst counts)
  6. 5-Stage Task Condition Comparison (Neutral vs Cognitive Load progression)
  7. Measurement Reliability Gauge
"""
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any

# Plotly Safe Transparent Color Constant
PLOTLY_TRANSPARENT = "rgba(0,0,0,0)"

# Brand Palette: Scientific Editorial
NAVY = "#0F172A"
NAVY_LIGHT = "#1E293B"
SLATE = "#475569"
SLATE_LIGHT = "#94A3B8"
BORDER = "#E2E8F0"
BG_PLOT = "rgba(248, 250, 252, 0.85)"
INDIGO = "#4F46E5"
INDIGO_SOFT = "rgba(79, 70, 229, 0.12)"
AMBER = "#D97706"
AMBER_SOFT = "rgba(217, 119, 6, 0.15)"
EMERALD = "#059669"
EMERALD_SOFT = "rgba(5, 150, 105, 0.15)"
ROSE = "#E11D48"
SKY = "#0284C7"


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Guards chart properties against None, NaN, Inf, and malformed types."""
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def create_keystroke_rhythm_chart(
    keystroke_events: List[Dict],
    baseline_mean: float = 180.0,
    baseline_std: float = 40.0
) -> go.Figure:
    """Full-width interactive line + scatter chart plotting the sequence
    of keystrokes against inter-key intervals (ms).
    
    Shows:
      - Individual keystroke intervals connected by cadence line
      - Baseline reference range (mean ± 1 std dev)
      - Notable hesitation spikes (intervals > mean + 2 std dev)
      - Interactive hover tooltips (Keystroke #, Interval, Classification)
    """
    b_mean = max(50.0, _safe_float(baseline_mean, 180.0))
    b_std = max(10.0, _safe_float(baseline_std, 35.0))

    if not keystroke_events or not isinstance(keystroke_events, list):
        # Fallback synthetic pattern for preview/empty states
        n = 100
        rng = np.random.default_rng(42)
        ikt = rng.normal(b_mean, b_std, n).clip(min=50.0).tolist()
        for idx in [20, 45, 72, 88]:
            if idx < n:
                ikt[idx] = b_mean + (b_std * 2.8) + rng.uniform(40, 150)
        is_bksp = [1 if i in [21, 46, 73] else 0 for i in range(n)]
    else:
        ikt = [max(10.0, _safe_float(e.get("interKeyMs", 0), b_mean)) for e in keystroke_events]
        is_bksp = [1 if int(e.get("isBackspace", 0)) == 1 else 0 for e in keystroke_events]
        n = len(ikt)
        if n == 0:
            ikt = [b_mean] * 20
            is_bksp = [0] * 20
            n = 20

    x_vals = list(range(1, n + 1))
    upper_threshold = b_mean + (2.0 * b_std)

    classifications = []
    marker_colors = []
    marker_sizes = []
    for val, b in zip(ikt, is_bksp):
        if val >= upper_threshold:
            classifications.append("Notable Hesitation Spike (z ≥ 2.0)")
            marker_colors.append(ROSE)
            marker_sizes.append(9)
        elif b == 1:
            classifications.append("Correction (Backspace)")
            marker_colors.append(AMBER)
            marker_sizes.append(7)
        elif val > b_mean + b_std:
            classifications.append("Mild Hesitation")
            marker_colors.append(SLATE)
            marker_sizes.append(5)
        else:
            classifications.append("Normal Keystroke Cadence")
            marker_colors.append(INDIGO)
            marker_sizes.append(4.5)

    fig = go.Figure()

    # Baseline Range Band (mean - std to mean + std)
    fig.add_hrect(
        y0=max(0.0, b_mean - b_std),
        y1=b_mean + b_std,
        fillcolor="rgba(79, 70, 229, 0.08)",
        line_width=0,
        annotation_text="Personal Baseline Range (±1σ)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=INDIGO)
    )

    # Upper Hesitation Threshold Line
    fig.add_hline(
        y=upper_threshold,
        line_width=1.5,
        line_dash="dash",
        line_color=AMBER,
        annotation_text="Hesitation Spike Threshold (z ≥ 2.0)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=AMBER)
    )

    # Cadence Line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=ikt,
        mode="lines",
        line=dict(color="rgba(71, 85, 105, 0.40)", width=1.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Keystroke Markers with Rich Tooltip
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=ikt,
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color="#FFFFFF", width=1)
        ),
        text=classifications,
        hovertemplate="<b>Keystroke %{x}</b><br>Interval: <b>%{y:.0f} ms</b><br>Status: %{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Keystroke Rhythm Stream</b> (Sequence vs. Inter-Key Interval in ms)",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color=NAVY),
            x=0.01,
        ),
        xaxis=dict(
            title="Keystroke Sequence Position (1 → N)",
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        yaxis=dict(
            title="Milliseconds Between Keystrokes (ms)",
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=45, r=20, t=45, b=40),
        height=320,
    )
    return fig


def create_baseline_diverging_chart(indicators: Dict[str, Any]) -> go.Figure:
    """Large horizontal comparison chart where 100 = Personal Baseline Reference.
    
    Protects against None/empty indicators and ensures clean numeric indices.
    """
    if not isinstance(indicators, dict):
        indicators = {}

    categories = [
        "Typing Cadence (Speed)",
        "Pause Frequency",
        "Correction Activity",
        "Rhythm Variability",
        "Task Consistency"
    ]

    # Map indicators to baseline index (100 = normal baseline)
    spd_delta = _safe_float(indicators.get("speed_delta_pct"), 0.0)
    pause_delta = _safe_float(indicators.get("pause_delta_pct"), 0.0)
    corr_delta = _safe_float(indicators.get("correction_delta_pct"), 0.0)
    rhythm_delta = _safe_float(indicators.get("rhythm_delta_pct"), 0.0)
    cons_score = _safe_float(indicators.get("consistency_score"), 85.0)

    spd_idx = max(30.0, min(170.0, 100.0 - spd_delta))
    pause_idx = max(30.0, min(170.0, 100.0 + pause_delta))
    corr_idx = max(30.0, min(170.0, 100.0 + corr_delta))
    rhythm_idx = max(30.0, min(170.0, 100.0 + rhythm_delta))
    cons_idx = max(30.0, min(170.0, cons_score))

    session_values = [spd_idx, pause_idx, corr_idx, rhythm_idx, cons_idx]
    baseline_values = [100.0] * 5

    fig = go.Figure()

    # Personal Baseline (Reference Bar)
    fig.add_trace(go.Bar(
        y=categories,
        x=baseline_values,
        orientation="h",
        name="Personal Baseline (100)",
        marker=dict(color="#E2E8F0", line=dict(color="#CBD5E1", width=1)),
        text=["100 (Baseline)"] * 5,
        textposition="inside",
        textfont=dict(size=10, color=SLATE, weight=600),
        hovertemplate="Baseline Reference Point: 100<extra></extra>",
    ))

    # Session Measurement Bar
    colors = []
    for s in session_values:
        if abs(s - 100.0) <= 10.0:
            colors.append(EMERALD)
        elif s > 115.0 or s < 85.0:
            colors.append(AMBER)
        else:
            colors.append(INDIGO)

    fig.add_trace(go.Bar(
        y=categories,
        x=session_values,
        orientation="h",
        name="Current Assessment",
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{s:.0f}" for s in session_values],
        textposition="inside",
        textfont=dict(size=11, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{y}</b>: %{x:.0f} (Baseline = 100)<extra></extra>",
    ))

    # Vertical 100 Baseline Reference Line
    fig.add_vline(
        x=100,
        line_width=2,
        line_dash="solid",
        line_color=NAVY,
        annotation_text="100 Baseline Reference",
        annotation_position="top left",
        annotation_font=dict(size=9, color=NAVY, weight=600)
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Behavioral Profile vs. Personal Baseline (100 = Calibration Reference)</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=NAVY),
            x=0.01,
        ),
        xaxis=dict(
            title="Index Score (100 = Your Own Calibration Baseline)",
            range=[0, 170],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=NAVY, weight=600),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=10, r=20, t=50, b=40),
        height=270,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=SLATE)
        )
    )
    return fig


def create_typing_speed_comparison(
    baseline_wpm: float = 48.0,
    current_wpm: float = 38.0,
    baseline_ms: float = 180.0,
    current_ms: float = 230.0
) -> go.Figure:
    """Comparative chart showing Words Per Minute and average key interval shifts."""
    b_wpm = max(1.0, _safe_float(baseline_wpm, 45.0))
    c_wpm = max(1.0, _safe_float(current_wpm, 38.0))
    b_ms = max(10.0, _safe_float(baseline_ms, 180.0))
    c_ms = max(10.0, _safe_float(current_ms, 230.0))

    labels = ["Resting Baseline (P1)", "Diagnostic Assessment"]
    wpm_vals = [b_wpm, c_wpm]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=wpm_vals,
        marker=dict(color=[EMERALD, AMBER if c_wpm < b_wpm * 0.9 else INDIGO]),
        text=[f"{w:.1f} WPM" for w in wpm_vals],
        textposition="auto",
        textfont=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{x}</b><br>Typing Speed: %{y:.1f} WPM<extra></extra>",
        width=[0.45, 0.45]
    ))

    fig.update_layout(
        title=dict(
            text="<b>Typing Speed Comparison (WPM)</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=NAVY),
            x=0.01,
        ),
        yaxis=dict(
            title="Estimated Words Per Minute",
            range=[0, max(wpm_vals) * 1.3],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        xaxis=dict(
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=NAVY, weight=600),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=40, r=20, t=45, b=30),
        height=240,
        showlegend=False,
    )
    return fig


def create_pause_distribution_chart(diag_results: List[Dict]) -> go.Figure:
    """Visualizes the distribution of pause intervals across duration thresholds
    (<500ms, 500-1000ms, 1-2s, >2s).
    """
    if not diag_results or not isinstance(diag_results, list):
        diag_results = []

    p500 = sum(_safe_float(r.get("pauses_500ms", 0)) for r in diag_results)
    p1000 = sum(_safe_float(r.get("pauses_1000ms", 0)) for r in diag_results)
    p2000 = sum(_safe_float(r.get("pauses_2000ms", 0)) for r in diag_results)
    p4000 = sum(_safe_float(r.get("pauses_4000ms", 0)) for r in diag_results)

    cat_short = max(0, int(p500 - p1000))
    cat_medium = max(0, int(p1000 - p2000))
    cat_long = max(0, int(p2000 - p4000))
    cat_extreme = max(0, int(p4000))

    categories = [
        "Short Pauses<br>(500–1000 ms)",
        "Moderate Pauses<br>(1–2 sec)",
        "Long Pauses<br>(2–4 sec)",
        "Extended Pauses<br>(>4 sec)"
    ]
    counts = [cat_short, cat_medium, cat_long, cat_extreme]
    colors = [SLATE, SKY, AMBER, ROSE]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=counts,
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{c}" if c > 0 else "0" for c in counts],
        textposition="auto",
        textfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{x}</b><br>Detected Count: %{y}<extra></extra>",
        width=0.55
    ))

    fig.update_layout(
        title=dict(
            text="<b>Hesitation & Pause Breakdown by Duration</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=NAVY),
            x=0.01,
        ),
        yaxis=dict(
            title="Total Pauses Detected",
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        xaxis=dict(
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=NAVY, weight=600),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=40, r=20, t=45, b=45),
        height=240,
        showlegend=False,
    )
    return fig


def create_correction_activity_chart(
    baseline_rate: float = 0.04,
    diagnostic_rate: float = 0.07,
    burst_count: int = 0
) -> go.Figure:
    """Dedicated chart comparing Baseline Correction Rate vs Diagnostic Correction Rate,
    illustrating self-monitoring and error-recovery frequency.
    """
    b_rate = max(0.0, _safe_float(baseline_rate, 0.04) * 100.0)
    d_rate = max(0.0, _safe_float(diagnostic_rate, 0.06) * 100.0)
    bursts = max(0, int(_safe_float(burst_count, 0)))

    labels = ["Resting Baseline (P1)", "Diagnostic Assessment"]
    rates = [b_rate, d_rate]
    colors = [SLATE, AMBER if d_rate > b_rate * 1.3 else INDIGO]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=rates,
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{r:.1f}%" for r in rates],
        textposition="auto",
        textfont=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{x}</b><br>Correction Rate: %{y:.1f}%<extra></extra>",
        width=[0.45, 0.45]
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Correction Activity</b> ({bursts} consecutive correction burst{'s' if bursts!=1 else ''})",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=NAVY),
            x=0.01,
        ),
        yaxis=dict(
            title="Backspace Key Frequency (%)",
            range=[0, max(rates + [10.0]) * 1.3],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        xaxis=dict(
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=NAVY, weight=600),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=40, r=20, t=45, b=30),
        height=240,
        showlegend=False,
    )
    return fig


def create_passage_comparison_chart(
    passage_results: List[Dict],
    passages_meta: List[Dict]
) -> go.Figure:
    """Grouped multi-signal comparison across all completed passages (P1 to P5)
    to visually determine whether behavioral shifts repeat across conditions.
    """
    if not passage_results or not isinstance(passage_results, list):
        passage_results = []
    if not passages_meta or not isinstance(passages_meta, list):
        passages_meta = []

    n = min(len(passage_results), len(passages_meta))
    if n == 0:
        labels = ["P1: Baseline", "P2: Neutral", "P3: Load", "P4: Neutral", "P5: Load"]
        strain = [10.0, 15.0, 75.0, 20.0, 70.0]
        bksp = [3.5, 4.0, 8.2, 4.2, 7.8]
    else:
        labels = [f"P{p.get('id', i+1)}: {p.get('type', '').title()}" for i, p in enumerate(passages_meta[:n])]
        strain = [_safe_float(r.get("strain_score", 50.0)) for r in passage_results[:n]]
        bksp = [_safe_float(r.get("backspace_rate", 0.0)) * 100.0 for r in passage_results[:n]]

    fig = go.Figure()

    # Strain Score Bar
    fig.add_trace(go.Bar(
        x=labels,
        y=strain,
        name="Behavioral Strain Index",
        marker=dict(color=AMBER, line=dict(color="#FFFFFF", width=1)),
        text=[f"{s:.0f}" for s in strain],
        textposition="inside",
        textfont=dict(size=10, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{x}</b><br>Strain Index: %{y:.1f}/100<extra></extra>",
    ))

    # Backspace Rate Bar (scaled × 5 for comparison)
    fig.add_trace(go.Bar(
        x=labels,
        y=[b * 5.0 for b in bksp],
        name="Correction Rate (×5 visual scale)",
        marker=dict(color=INDIGO, line=dict(color="#FFFFFF", width=1)),
        text=[f"{b:.1f}%" for b in bksp],
        textposition="inside",
        textfont=dict(size=10, color="#FFFFFF", weight=700),
        hovertemplate="<b>%{x}</b><br>Correction Rate: %{text}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Cross-Condition Task Comparison (P1 Baseline → P2 Neutral → P3 Load → P4 Neutral → P5 Load)</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13.5, color=NAVY),
            x=0.01,
        ),
        xaxis=dict(
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=NAVY, weight=600),
        ),
        yaxis=dict(
            title="Metric Scale",
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=SLATE),
        ),
        paper_bgcolor=PLOTLY_TRANSPARENT,
        plot_bgcolor=BG_PLOT,
        margin=dict(l=40, r=20, t=50, b=35),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=SLATE)
        )
    )
    return fig


def create_reliability_gauge(reliability_score: float, tier: str) -> go.Figure:
    """Semicircular indicator for Measurement Reliability (0-100)."""
    score = max(0.0, min(100.0, _safe_float(reliability_score, 80.0)))
    tier_str = str(tier or "Moderate")

    color = EMERALD if score >= 75.0 else (AMBER if score >= 50.0 else ROSE)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix=" / 100", font=dict(family="Plus Jakarta Sans", size=26, color=NAVY, weight=800)),
        title=dict(text=f"<b>Measurement Reliability: {tier_str}</b>", font=dict(family="Plus Jakarta Sans", size=13, color=SLATE)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=SLATE_LIGHT, tickfont=dict(size=9, color=SLATE_LIGHT)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="#F1F5F9",
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color="rgba(225, 29, 72, 0.08)"),
                dict(range=[50, 75], color="rgba(217, 119, 6, 0.08)"),
                dict(range=[75, 100], color="rgba(5, 150, 105, 0.08)"),
            ],
            threshold=dict(
                line=dict(color=NAVY, width=2.5),
                thickness=0.75,
                value=score
            )
        )
    ))

    fig.update_layout(
        paper_bgcolor=PLOTLY_TRANSPARENT,
        margin=dict(l=30, r=30, t=35, b=10),
        height=180,
    )
    return fig
