"""
report.py
----------
Synthesizes 5-stage typing assessment data into an explainable, research-grade
behavioral analysis report.

Components generated:
  1. Overall Behavioral Strain Index (0-100) & Measurement Reliability (0-100)
  2. Session Quality Score (0-100) & factor audit checklist
  3. 5 Behavioral Indicator metrics (relative % changes from personal baseline)
  4. Structured "What We Observed" narrative cards
  5. Scientific context, alternative factors, and explicit non-clinical limitations
  6. Multi-format exports: standalone print-ready HTML report & clean text report
"""
import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Any, Tuple

from reliability import compute_session_reliability
from interpretation import (
    compute_behavioral_indicators,
    generate_behavioral_cards,
    interpret_behavioral_strain
)

BACKSPACE_FLAG_RATE = 0.065
ERROR_FLAG_RATE = 0.050
BASELINE_DEV_FLAG = 0.40
RHYTHM_CV_RISE_RATIO = 1.20


def build_features_dataframe_from_events(events: list, passage_id: int, session_id: str) -> pd.DataFrame:
    """Converts the raw JS-captured events from the browser widget into
    the canonical feature-row shape used throughout the pipeline.
    """
    if not events:
        raise ValueError(
            f"No keystroke events were captured for passage {passage_id}. "
            "Please ensure typing occurred in the widget before submitting."
        )
    rows = []
    for i, e in enumerate(events):
        rows.append({
            "key_index": i,
            "inter_key_ms": max(float(e.get("interKeyMs", 0.0)), 0.0),
            "dwell_ms": max(float(e.get("dwellMs", 0.0)), 0.0),
            "is_backspace": int(e.get("isBackspace", 0)),
            "is_error": int(e.get("isError", 0)),
        })
    df = pd.DataFrame(rows)
    df["session_id"] = f"{session_id}_p{passage_id}"
    return df


def calibrate_baseline_from_passage_1(passage_1_df: pd.DataFrame) -> Tuple[float, float, np.ndarray, dict]:
    """Establishes this session's personal baseline from Passage 1 (natural calibration).
    Returns (mean_ikt, std_ikt, raw_ikt_array, baseline_stats_dict).
    """
    mean_val = float(passage_1_df["inter_key_ms"].mean())
    std_val = float(passage_1_df["inter_key_ms"].std()) or 25.0
    raw_arr = passage_1_df["inter_key_ms"].to_numpy(dtype=float)

    bksp_rate = float(passage_1_df["is_backspace"].mean())
    err_rate = float(passage_1_df["is_error"].mean())
    cv_val = float(std_val / (mean_val + 1e-6))
    pauses_500 = int(np.sum(raw_arr >= 500))
    pause_freq = float((pauses_500 / max(len(raw_arr), 1)) * 100)

    baseline_dict = {
        "inter_key_mean_ms": round(mean_val, 1),
        "inter_key_std_ms": round(std_val, 1),
        "inter_key_cv": round(cv_val, 3),
        "backspace_rate": round(bksp_rate, 4),
        "error_rate": round(err_rate, 4),
        "pause_frequency_pct": round(pause_freq, 1),
        "n_keystrokes": len(raw_arr),
    }

    return mean_val, max(std_val, 15.0), raw_arr, baseline_dict


def _evidence_for_passage(
    result: dict,
    baseline_stats: dict,
    is_calibration: bool,
    condition_name: str
) -> List[str]:
    """Builds concrete, numeric observations for a single passage."""
    notes = []
    if is_calibration:
        notes.append("Resting baseline established: provides individual reference for timing and rhythm.")
        return notes

    dev = result.get("baseline_deviation", 0.0)
    if dev >= BASELINE_DEV_FLAG:
        notes.append(
            f"Typing cadence slowed by {dev:+.2f} baseline standard deviations relative to the calibration baseline."
        )
    elif dev <= -BASELINE_DEV_FLAG:
        notes.append(
            f"Typing cadence accelerated by {abs(dev):.2f} baseline standard deviations relative to calibration."
        )

    b_bksp = baseline_stats.get("backspace_rate", 0.04)
    bksp = result.get("backspace_rate", 0.0)
    bursts = result.get("correction_bursts", 0)
    if bksp >= BACKSPACE_FLAG_RATE or bksp > (b_bksp * 1.5):
        burst_msg = f" with {bursts} consecutive backspace burst(s)" if bursts > 0 else ""
        notes.append(f"Correction rate elevated at {bksp*100:.1f}% (vs {b_bksp*100:.1f}% baseline){burst_msg}.")

    spikes = result.get("hesitation_spikes", 0)
    near_err = result.get("hesitation_near_error", 0)
    if spikes > 0:
        s = f"{spikes} outlier pause(s) detected mid-passage (z-score >= 2.0)"
        if near_err > 0:
            s += f", with {near_err} occurring adjacent to character edits"
        notes.append(s + ".")

    b_cv = baseline_stats.get("inter_key_cv", 0.35)
    cv = result.get("inter_key_cv", result.get("rhythm_cv", 0.35))
    if cv >= b_cv * RHYTHM_CV_RISE_RATIO:
        notes.append(
            f"Rhythm variability increased (CV {cv:.2f} vs {b_cv:.2f} baseline), reflecting less uniform cadence."
        )

    p_val = result.get("rhythm_shift_p")
    if p_val is not None:
        if result.get("rhythm_shift_significant"):
            notes.append(
                f"Mann-Whitney U test confirmed significant rhythm shift from calibration (p = {p_val:.4f})."
            )
        else:
            notes.append(
                f"Mann-Whitney U test showed rhythm timing remained statistically consistent with calibration (p = {p_val:.4f})."
            )

    if not notes:
        notes.append(f"Typing dynamics under {condition_name} remained closely aligned with baseline parameters.")

    return notes


def aggregate_report(
    passage_results: List[Dict],
    passages_meta: List[Dict],
    participant_label: str = "Participant"
) -> Dict[str, Any]:
    """Aggregates all 5 passage results into the master report data structure."""
    if not passage_results or len(passage_results) < 2:
        raise ValueError("aggregate_report requires at least a calibration passage and one diagnostic passage.")

    # Passage 1 is calibration; diagnostic are passages 2..N
    calib_result = passage_results[0]
    diag_results = passage_results[1:]

    diag_scores = [r.get("strain_score", 50.0) for r in diag_results]
    overall_strain_index = round(float(np.mean(diag_scores)), 1)

    # Reliability calculation
    rel_data = compute_session_reliability(passage_results, passages_meta)

    # Extract baseline stats dictionary
    baseline_stats = {
        "inter_key_mean_ms": calib_result.get("inter_key_mean_ms", calib_result.get("avg_typing_speed_ms", 180.0)),
        "inter_key_cv": calib_result.get("inter_key_cv", calib_result.get("rhythm_cv", 0.35)),
        "backspace_rate": calib_result.get("backspace_rate", 0.04),
        "error_rate": calib_result.get("error_rate", 0.03),
        "pause_frequency_pct": calib_result.get("pause_frequency_pct", 5.0),
        "n_keystrokes": calib_result.get("n_keystrokes", 100),
    }

    # Behavioral Indicators
    indicators = compute_behavioral_indicators(passage_results)

    # Narrative Cards
    observed_cards = generate_behavioral_cards(indicators, diag_results)

    # Interpretation & Limitations (Multi-Signal Combinatorial Engine)
    interp = interpret_behavioral_strain(
        overall_strain_index,
        rel_data["reliability_tier"],
        indicators,
        diag_results=diag_results,
        rel_data=rel_data,
        baseline_stats=baseline_stats
    )

    # Enrich passage results with per-passage evidence notes
    enriched_results = []
    for i, r in enumerate(passage_results):
        r_copy = dict(r)
        meta = passages_meta[i] if i < len(passages_meta) else {"condition": "Unknown", "title": f"Passage {i+1}"}
        r_copy["evidence"] = _evidence_for_passage(
            r,
            baseline_stats,
            is_calibration=(i == 0),
            condition_name=meta.get("condition", "Standard")
        )
        enriched_results.append(r_copy)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate Exports
    txt_report = generate_text_report(
        participant_label=participant_label,
        timestamp=now_str,
        strain_index=overall_strain_index,
        rel_data=rel_data,
        indicators=indicators,
        observed_cards=observed_cards,
        interp=interp,
        enriched_results=enriched_results,
        passages_meta=passages_meta
    )

    html_report = generate_html_report(
        participant_label=participant_label,
        timestamp=now_str,
        strain_index=overall_strain_index,
        rel_data=rel_data,
        indicators=indicators,
        observed_cards=observed_cards,
        interp=interp,
        enriched_results=enriched_results,
        passages_meta=passages_meta
    )

    return {
        "participant_label": participant_label,
        "timestamp": now_str,
        "strain_index": overall_strain_index,
        "tier_label": interp["tier_label"],
        "reliability": rel_data,
        "indicators": indicators,
        "observed_cards": observed_cards,
        "interpretation": interp,
        "passage_results": enriched_results,
        "txt_report": txt_report,
        "html_report": html_report,
        "is_conclusive": rel_data["is_conclusive"],
    }


def generate_text_report(
    participant_label: str,
    timestamp: str,
    strain_index: float,
    rel_data: dict,
    indicators: dict,
    observed_cards: list,
    interp: dict,
    enriched_results: list,
    passages_meta: list
) -> str:
    """Produces a clean, human-readable plain text report."""
    lines = [
        "=" * 72,
        "MINDTYPE — TYPING BEHAVIOR & COGNITIVE STRAIN REPORT",
        "Research Prototype for Keystroke Dynamics Analysis",
        "=" * 72,
        f"Participant:        {participant_label}",
        f"Assessment Date:    {timestamp}",
        f"Session Protocol:   5-Stage Alternating Assessment (Calibration -> Neutral -> Load)",
        "-" * 72,
        "",
        "1. OVERALL SESSION ASSESSMENT",
        f"   Behavioral Strain Index:  {strain_index:.1f} / 100 ({interp['tier_label']})",
        f"   Measurement Reliability:  {rel_data['reliability_score']:.1f} / 100 ({rel_data['reliability_tier']})",
        f"   Session Quality Score:    {rel_data['session_quality_score']:.1f} / 100 ({rel_data['session_quality_tier']})",
        "",
        f"   Session Status: {'Conclusive' if rel_data['is_conclusive'] else 'INCONCLUSIVE'}",
    ]

    if not rel_data["is_conclusive"]:
        lines.append("   Notice: Reliability is below conclusive threshold. Results are indicative only.")
        for r in rel_data.get("inconclusive_reasons", []):
            lines.append(f"     • {r}")

    lines.extend([
        "",
        "2. CORE BEHAVIORAL INDICATORS (Relative to Personal Baseline)",
        f"   • Typing Speed Shift:        {indicators['speed_delta_pct']:+.1f}% ({indicators['speed_direction']})",
        f"   • Pause Frequency Shift:     {indicators['pause_delta_pct']:+.1f}% ({indicators['pause_direction']})",
        f"   • Rhythm Variability Shift:  {indicators['rhythm_delta_pct']:+.1f}% ({indicators['rhythm_direction']})",
        f"   • Correction Rate Shift:     {indicators['correction_delta_pct']:+.1f}% ({indicators['correction_direction']})",
        f"   • Cross-Passage Consistency: {indicators['consistency_score']:.0f} / 100",
        "",
        "3. WHAT WE OBSERVED",
    ])

    for card in observed_cards:
        lines.append(f"   [{card['dimension']}] ({card['badge']})")
        lines.append(f"     {card['narrative']}")
        lines.append("")

    lines.extend([
        "4. PASSAGE-BY-PASSAGE BREAKDOWN",
    ])

    for i, r in enumerate(enriched_results):
        meta = passages_meta[i] if i < len(passages_meta) else {}
        lines.append(f"   Passage {i+1}: {meta.get('title', 'Task')}")
        lines.append(f"     Condition:       {meta.get('condition', 'Standard')}")
        lines.append(f"     Strain Score:    {r.get('strain_score', 0):.1f}/100")
        lines.append(f"     Avg Key Interval:{r.get('avg_typing_speed_ms', 0):.1f} ms")
        lines.append(f"     Backspace Rate:  {r.get('backspace_rate', 0)*100:.1f}%")
        lines.append("     Evidence:")
        for ev in r.get("evidence", []):
            lines.append(f"       • {ev}")
        lines.append("")

    lines.extend([
        "5. INTERPRETATION & SCIENTIFIC CAUTION",
        f"   {interp['summary']}",
        "",
        "   Alternative Explanations for Typing Shifts:",
        f"{interp['cautious_context']}",
        "",
        "6. IMPORTANT SCIENTIFIC LIMITATIONS",
        f"{interp['limitations']}",
        "",
        "7. RECOMMENDED NEXT STEP",
        f"   {interp['next_step']}",
        "=" * 72,
    ])

    return "\n".join(lines)


def generate_html_report(
    participant_label: str,
    timestamp: str,
    strain_index: float,
    rel_data: dict,
    indicators: dict,
    observed_cards: list,
    interp: dict,
    enriched_results: list,
    passages_meta: list
) -> str:
    """Produces an elegant, self-contained standalone HTML report formatted with
    Plus Jakarta Sans typography, suitable for printing to PDF or saving.
    """
    total_keystrokes = sum(r.get("n_keystrokes", 0) for r in enriched_results)
    card_htmls = []
    for c in observed_cards:
        card_htmls.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="card-title"><span class="icon">{c['icon']}</span> {c['dimension']}</span>
            <span class="badge {c['status']}">{c['badge']}</span>
          </div>
          <p class="card-text">{c['narrative']}</p>
        </div>
        """)

    passage_rows = []
    for i, r in enumerate(enriched_results):
        meta = passages_meta[i] if i < len(passages_meta) else {}
        evidence_items = "".join([f"<li>{ev}</li>" for ev in r.get("evidence", [])])
        passage_rows.append(f"""
        <tr>
          <td><strong>P{i+1}</strong></td>
          <td>
            <div style="font-weight:700;">{meta.get('title', 'Task')}</div>
            <div style="font-size:12px; color:#64748B;">{meta.get('condition', 'Standard')}</div>
          </td>
          <td><strong>{r.get('strain_score', 0):.1f}</strong> / 100</td>
          <td>{r.get('avg_typing_speed_ms', 0):.0f} ms</td>
          <td>{r.get('backspace_rate', 0)*100:.1f}%</td>
          <td><ul class="evidence-list">{evidence_items}</ul></td>
        </tr>
        """)

    inconclusive_banner = ""
    if not rel_data["is_conclusive"]:
        reasons = "".join([f"<li>{r}</li>" for r in rel_data.get("inconclusive_reasons", [])])
        inconclusive_banner = f"""
        <div class="alert-box">
          <h3>&#9888; Inconclusive Session Notice</h3>
          <p>This assessment contains data irregularities or interruptions that lower measurement reliability below research standards:</p>
          <ul>{reasons}</ul>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MindType Report — {participant_label}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  @page {{
    margin: 1.5cm;
    size: auto;
  }}
  body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0; padding: 36px 24px;
    background: #F8FAFC; color: #0F172A;
    line-height: 1.6; -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 920px; margin: 0 auto; background: #FFFFFF;
    border-radius: 20px; border: 1px solid #E2E8F0; padding: 44px;
    box-shadow: 0 4px 24px rgba(15,23,42,0.06);
  }}
  .header {{
    border-bottom: 2px solid #F1F5F9; padding-bottom: 24px; margin-bottom: 28px;
    display: flex; justify-content: space-between; align-items: flex-start;
  }}
  .logo {{
    font-size: 24px; font-weight: 800; letter-spacing: -0.02em; color: #0F172A;
    display: flex; align-items: center; gap: 8px;
  }}
  .sub {{ font-size: 13px; color: #64748B; margin-top: 4px; }}
  .meta {{ text-align: right; font-size: 13px; color: #64748B; }}
  .meta strong {{ color: #0F172A; }}

  /* Editorial Hero Layout (Replaces Bento UI) */
  .editorial-hero {{
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 20px;
    margin-bottom: 30px;
  }}
  .hero-main {{
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #FFFFFF;
    border-radius: 18px;
    padding: 28px 30px;
  }}
  .hero-eyebrow {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #F59E0B; margin-bottom: 8px;
  }}
  .hero-score-row {{
    display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;
  }}
  .hero-score {{
    font-size: 48px; font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: #FFFFFF;
  }}
  .hero-denom {{
    font-size: 18px; color: #94A3B8; font-weight: 600;
  }}
  .hero-badge {{
    display: inline-block; font-size: 12px; font-weight: 700;
    padding: 4px 12px; border-radius: 999px; background: rgba(245, 158, 11, 0.2);
    color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4);
    margin-bottom: 12px;
  }}
  .hero-narrative {{
    font-size: 13.5px; color: #CBD5E1; line-height: 1.55; margin: 0;
  }}
  .hero-audit {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .audit-block {{ margin-bottom: 14px; }}
  .audit-label {{
    font-size: 11.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748B; margin-bottom: 4px;
  }}
  .audit-val {{
    font-size: 24px; font-weight: 800; color: #0F172A;
  }}
  .audit-sub {{
    font-size: 12.5px; color: #475569; margin-top: 2px;
  }}

  /* Section Titles */
  .section-title {{
    font-size: 17px; font-weight: 800; color: #0F172A;
    margin: 34px 0 16px 0; letter-spacing: -0.01em;
    border-left: 4px solid #4F46E5; padding-left: 12px;
  }}

  /* Baseline Comparison Table */
  .baseline-table {{
    width: 100%; border-collapse: collapse; margin-bottom: 28px;
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden;
  }}
  .baseline-table th {{
    background: #F8FAFC; text-align: left; padding: 12px 16px;
    font-size: 11.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748B; border-bottom: 1px solid #E2E8F0;
  }}
  .baseline-table td {{
    padding: 12px 16px; border-bottom: 1px solid #F1F5F9; font-size: 13px;
  }}
  .delta-pill {{
    display: inline-block; font-size: 11.5px; font-weight: 700;
    padding: 3px 8px; border-radius: 6px;
  }}
  .delta-pill.stable {{ background: #DCFCE7; color: #166534; }}
  .delta-pill.elevated {{ background: #FEF3C7; color: #92400E; }}

  /* Cards Grid */
  .cards-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 28px; }}
  .card {{
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    border-left: 4px solid #E2E8F0;
  }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .card-title {{ font-size: 14px; font-weight: 700; color: #0F172A; }}
  .badge {{ font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; background: #F1F5F9; color: #475569; }}
  .badge.slower, .badge.elevated, .badge.variable {{ background: #FEF3C7; color: #92400E; }}
  .badge.stable, .badge.high {{ background: #DCFCE7; color: #166534; }}
  .card-text {{ font-size: 13px; color: #475569; margin: 0; line-height: 1.55; }}

  /* Passage Table */
  table.passage-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
  table.passage-table th {{
    background: #F8FAFC; text-align: left; padding: 10px 12px;
    font-size: 11.5px; font-weight: 700; text-transform: uppercase;
    color: #64748B; border-bottom: 1px solid #E2E8F0;
  }}
  table.passage-table td {{ padding: 12px; border-bottom: 1px solid #F1F5F9; vertical-align: top; }}
  .evidence-list {{ margin: 0; padding-left: 18px; color: #64748B; font-size: 12px; }}

  /* Alerts & Callouts */
  .alert-box {{ background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 14px; padding: 16px 20px; margin-bottom: 24px; color: #92400E; }}
  .alert-box h3 {{ margin: 0 0 6px 0; font-size: 15px; font-weight: 700; }}
  .callout {{ background: #F1F5F9; border-radius: 14px; padding: 18px 22px; margin-top: 14px; font-size: 13px; color: #334155; line-height: 1.6; }}
  .callout strong {{ color: #0F172A; }}
  .disclaimer {{ margin-top: 36px; padding-top: 20px; border-top: 1px solid #E2E8F0; font-size: 11.5px; color: #94A3B8; text-align: center; }}

  @media print {{
    body {{ background: #FFFFFF; padding: 0; }}
    .container {{ border: none; box-shadow: none; padding: 0; max-width: 100%; }}
    .editorial-hero {{ page-break-inside: avoid; }}
    .cards-grid {{ page-break-inside: avoid; }}
    table.passage-table {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div class="logo">&#129504; MINDTYPE</div>
      <div class="sub">Typing Behavior & Cognitive Strain Research Analysis</div>
    </div>
    <div class="meta">
      <div>Participant: <strong>{participant_label}</strong></div>
      <div>Date: <strong>{timestamp}</strong></div>
      <div>Protocol: <strong>5-Stage Alternating Protocol</strong></div>
    </div>
  </div>

  {inconclusive_banner}

  <!-- Editorial Hero Layout -->
  <div class="editorial-hero">
    <div class="hero-main">
      <div class="hero-eyebrow">Overall Assessment</div>
      <div class="hero-score-row">
        <span class="hero-score">{strain_index:.1f}</span>
        <span class="hero-denom">/ 100</span>
      </div>
      <div class="hero-badge">{interp['tier_label']}</div>
      <p class="hero-narrative">{interp['summary']}</p>
    </div>
    <div class="hero-audit">
      <div class="audit-block">
        <div class="audit-label">Measurement Reliability</div>
        <div class="audit-val">{rel_data['reliability_score']:.0f}<span style="font-size:16px; color:#64748B;">/100</span></div>
        <div class="audit-sub">{rel_data['reliability_tier']} Confidence ({'Conclusive' if rel_data['is_conclusive'] else 'Inconclusive'})</div>
      </div>
      <div class="audit-block" style="margin-bottom:0;">
        <div class="audit-label">Session Quality & Focus</div>
        <div class="audit-val">{rel_data['session_quality_score']:.0f}<span style="font-size:16px; color:#64748B;">/100</span></div>
        <div class="audit-sub">{rel_data['session_quality_tier']} Continuity ({total_keystrokes} total keystrokes)</div>
      </div>
    </div>
  </div>

  <!-- Comparative Baseline Dynamics Table -->
  <div class="section-title">Baseline vs. Diagnostic Comparative Dynamics</div>
  <table class="baseline-table">
    <thead>
      <tr>
        <th>Behavioral Dimension</th>
        <th>Shift from Baseline</th>
        <th>Observed Cadence Direction</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Typing Cadence (Gap between keys)</strong></td>
        <td><span class="delta-pill {'elevated' if abs(indicators['speed_delta_pct']) > 10 else 'stable'}">{indicators['speed_delta_pct']:+.1f}%</span></td>
        <td>{indicators['speed_direction']}</td>
      </tr>
      <tr>
        <td><strong>Pause Frequency (Hesitation rate)</strong></td>
        <td><span class="delta-pill {'elevated' if indicators['pause_delta_pct'] > 15 else 'stable'}">{indicators['pause_delta_pct']:+.1f}%</span></td>
        <td>{indicators['pause_direction']}</td>
      </tr>
      <tr>
        <td><strong>Rhythm Uniformity (CV variability)</strong></td>
        <td><span class="delta-pill {'elevated' if indicators['rhythm_delta_pct'] > 15 else 'stable'}">{indicators['rhythm_delta_pct']:+.1f}%</span></td>
        <td>{indicators['rhythm_direction']}</td>
      </tr>
      <tr>
        <td><strong>Correction Dynamics (Backspace usage)</strong></td>
        <td><span class="delta-pill {'elevated' if indicators['correction_delta_pct'] > 15 else 'stable'}">{indicators['correction_delta_pct']:+.1f}%</span></td>
        <td>{indicators['correction_direction']}</td>
      </tr>
      <tr>
        <td><strong>Cross-Passage Consistency</strong></td>
        <td><span class="delta-pill {'stable' if indicators['consistency_score'] >= 60 else 'elevated'}">{indicators['consistency_score']:.0f} / 100</span></td>
        <td>{'Uniform pace maintained' if indicators['consistency_score'] >= 60 else 'Fluctuating across tasks'}</td>
      </tr>
    </tbody>
  </table>

  <div class="section-title">What We Observed Across 5 Behavioral Dimensions</div>
  <div class="cards-grid">
    {''.join(card_htmls)}
  </div>

  <div class="section-title">5-Stage Passage-by-Passage Analysis</div>
  <table class="passage-table">
    <thead>
      <tr>
        <th>Stage</th>
        <th>Task Condition</th>
        <th>Strain Score</th>
        <th>Speed</th>
        <th>Backspaces</th>
        <th>Observed Evidence</th>
      </tr>
    </thead>
    <tbody>
      {''.join(passage_rows)}
    </tbody>
  </table>

  <div class="section-title">Interpretation & Scientific Caution</div>
  <div class="callout">
    <strong>Alternative Factors Behind Typing Shifts:</strong><br>
    {interp['cautious_context'].replace(chr(10), '<br>')}
  </div>

  <div class="section-title">Important Scientific Limitations</div>
  <div class="callout" style="background:#FFFBEB; border:1px solid #FEF3C7; color:#92400E;">
    {interp['limitations'].replace(chr(10), '<br>')}
  </div>

  <div class="section-title">Recommended Next Step</div>
  <p style="font-size: 13.5px; color: #475569;">{interp['next_step']}</p>

  <div class="disclaimer">
    MindType is an experimental research prototype for typing behavior analysis. It does not provide medical, psychiatric, or psychological diagnoses.
  </div>
</div>
</body>
</html>
"""
    return html
