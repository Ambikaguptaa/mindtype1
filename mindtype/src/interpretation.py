"""
interpretation.py
------------------
Multi-Signal Behavioral Interpretation Engine for MindType.

Translates multi-dimensional keystroke dynamics, browser focus telemetry,
robust statistical distributions, and personal baseline deltas into structured,
combinatorial, scientifically cautious behavioral insights.

CORE PRINCIPLES:
  - NEVER claims medical or psychiatric diagnosis (NO depression, anxiety, ADHD, burnout).
  - Uses direct, qualified language:
      "Your typing pattern is consistent with..."
      "Your typing behavior may suggest..."
      "The observed pattern may be associated with..."
      "This session shows behavioral signals commonly associated with..."
      "Possible interpretation: ..."
  - Combinatorial multi-signal engine: Combines 6 evidence groups:
      A. Speed (WPM, speed drift, inter-key interval)
      B. Hesitation (long pauses, pause frequency, hesitation spikes, longest pause)
      C. Corrections (backspaces, correction bursts, error clusters)
      D. Rhythm (rhythm CV, IKI variability, distribution shifts)
      E. Focus & Session Quality (tab switches, window blurs, paste attempts, inactivity)
      F. Cross-Passage Consistency (Neutral vs Cognitive Load repeated patterns)
  - Distinguishes "Cognitive Strain" from "Distraction / Interruption" from "Fatigue-like" from "Stable".
  - Clarifies that ML/statistical libraries (scikit-learn, TensorFlow, tsfresh, SciPy)
    extract timing distributions and temporal features, NOT subjective mental states.
"""
from typing import Dict, List, Any
import numpy as np


def compute_behavioral_indicators(passage_results: List[Dict]) -> Dict[str, Any]:
    """Extracts aggregate comparative shifts across the 5 behavioral dimensions
    relative to the Passage 1 calibration baseline.
    """
    if not passage_results or len(passage_results) < 2:
        return {
            "speed_delta_pct": 0.0,
            "pause_delta_pct": 0.0,
            "rhythm_delta_pct": 0.0,
            "correction_delta_pct": 0.0,
            "consistency_score": 100.0,
            "speed_direction": "stable",
            "pause_direction": "stable",
            "rhythm_direction": "stable",
            "correction_direction": "stable",
        }

    calib = passage_results[0]
    diag = passage_results[1:]

    # Calibration baseline values
    b_speed = float(calib.get("inter_key_mean_ms", calib.get("avg_typing_speed_ms", 180.0)))
    b_pause = float(calib.get("pause_frequency_pct", 5.0))
    b_cv = float(calib.get("inter_key_cv", calib.get("rhythm_cv", 0.35)))
    b_bksp = float(calib.get("backspace_rate", 0.04))

    # Diagnostic mean values
    d_speed = float(np.mean([r.get("inter_key_mean_ms", r.get("avg_typing_speed_ms", b_speed)) for r in diag]))
    d_pause = float(np.mean([r.get("pause_frequency_pct", b_pause) for r in diag]))
    d_cv = float(np.mean([r.get("inter_key_cv", r.get("rhythm_cv", b_cv)) for r in diag]))
    d_bksp = float(np.mean([r.get("backspace_rate", b_bksp) for r in diag]))

    # Percentage deltas (higher inter_key_ms means SLOWER typing cadence)
    speed_delta = round(((d_speed - b_speed) / max(b_speed, 1.0)) * 100.0, 1)
    pause_delta = round(((d_pause - b_pause) / max(b_pause, 0.1)) * 100.0, 1)
    rhythm_delta = round(((d_cv - b_cv) / max(b_cv, 0.05)) * 100.0, 1)
    correction_delta = round(((d_bksp - b_bksp) / max(b_bksp, 0.01)) * 100.0, 1)

    # Cross-Passage Consistency (spread of strain scores across diagnostic tasks)
    diag_scores = [float(r.get("strain_score", 50.0)) for r in diag]
    score_std = float(np.std(diag_scores)) if len(diag_scores) > 1 else 0.0
    consistency_pct = max(0.0, min(100.0, 100.0 - (score_std * 2.5)))

    # Direction labels
    if speed_delta > 8.0:
        speed_dir = f"↓ {abs(speed_delta):.1f}% Slower"
    elif speed_delta < -8.0:
        speed_dir = f"↑ {abs(speed_delta):.1f}% Faster"
    else:
        speed_dir = "Stable (±8%)"

    pause_dir = f"↑ {abs(pause_delta):.1f}% Elevated" if pause_delta > 15.0 else ("↓ Reduced" if pause_delta < -15.0 else "Stable")
    rhythm_dir = f"↑ {abs(rhythm_delta):.1f}% More Variable" if rhythm_delta > 12.0 else ("Steady Cadence" if rhythm_delta < -12.0 else "Stable")
    corr_dir = f"↑ {abs(correction_delta):.1f}% Elevated" if correction_delta > 15.0 else ("Low Edits" if correction_delta < -15.0 else "Consistent")

    return {
        "speed_delta_pct": speed_delta,
        "pause_delta_pct": pause_delta,
        "rhythm_delta_pct": rhythm_delta,
        "correction_delta_pct": correction_delta,
        "consistency_score": round(consistency_pct, 1),
        "speed_direction": speed_dir,
        "pause_direction": pause_dir,
        "rhythm_direction": rhythm_dir,
        "correction_direction": corr_dir,
    }


def generate_behavioral_cards(indicators: Dict[str, Any], diag_results: List[Dict]) -> List[Dict[str, str]]:
    """Generates structured qualitative observation cards for the 5 behavioral dimensions."""
    cards = []
    if not isinstance(indicators, dict):
        indicators = {}
    if not isinstance(diag_results, list):
        diag_results = []

    # 1. Typing Speed
    spd_val = indicators.get("speed_delta_pct", 0.0)
    if spd_val > 10.0:
        spd_text = (
            f"Your typing cadence widened by approximately {abs(spd_val):.1f}% relative to calibration. "
            "A longer inter-key interval under demanding task conditions commonly reflects increased deliberate "
            "cognitive planning and cautious motor execution."
        )
        spd_badge = f"↓ {abs(spd_val):.1f}% Slower"
        spd_status = "slower"
    elif spd_val < -10.0:
        spd_text = (
            f"Your typing cadence was approximately {abs(spd_val):.1f}% faster than your baseline. "
            "Accelerated typing often indicates heightened task momentum, familiar vocabulary, or compensatory pacing."
        )
        spd_badge = f"↑ {abs(spd_val):.1f}% Faster"
        spd_status = "faster"
    else:
        spd_text = (
            f"Your typing cadence remained closely aligned with your personal baseline (within {abs(spd_val):.1f}%). "
            "Pacing was stable across changing task conditions."
        )
        spd_badge = "Stable (±10%)"
        spd_status = "stable"

    cards.append({
        "dimension": "Typing Speed & Cadence",
        "icon": "⏱",
        "badge": spd_badge,
        "status": spd_status,
        "narrative": spd_text,
    })

    # 2. Hesitation & Pauses
    pause_val = indicators.get("pause_delta_pct", 0.0)
    total_spikes = sum(int(r.get("hesitation_spikes", 0)) for r in diag_results)
    near_errors = sum(int(r.get("hesitation_near_error", 0)) for r in diag_results)

    if pause_val > 20.0 or total_spikes >= 3:
        p_detail = f" with {total_spikes} mid-passage hesitation spikes" if total_spikes > 0 else ""
        if near_errors > 0:
            p_detail += f" ({near_errors} occurring immediately adjacent to character edits)"
        pause_text = (
            f"Pause frequency was {abs(pause_val):.1f}% above your calibrated baseline{p_detail}. "
            "Hesitation spikes cluster when individuals pause to plan difficult words or re-orient motor focus."
        )
        p_badge = f"↑ {abs(pause_val):.1f}% Pauses"
        p_status = "elevated"
    else:
        pause_text = (
            "Pauses between words and keys remained uniform with low hesitation spike activity. "
            "Typing continuity was maintained smoothly without abnormal pauses."
        )
        p_badge = "Steady Pacing"
        p_status = "stable"

    cards.append({
        "dimension": "Hesitation & Pauses",
        "icon": "⏸",
        "badge": p_badge,
        "status": p_status,
        "narrative": pause_text,
    })

    # 3. Corrections & Backspaces
    corr_val = indicators.get("correction_delta_pct", 0.0)
    total_bursts = sum(int(r.get("correction_bursts", 0)) for r in diag_results)
    if corr_val > 25.0 or total_bursts >= 2:
        corr_text = (
            f"Correction activity increased by {abs(corr_val):.1f}% compared with your baseline, "
            f"including {total_bursts} consecutive backspace burst(s). "
            "Elevated correction activity may indicate higher self-monitoring, typing difficulty, or motor error recovery."
        )
        c_badge = f"↑ {abs(corr_val):.1f}% Corrections"
        c_status = "elevated"
    else:
        corr_text = (
            f"Correction frequency stayed low and consistent with baseline ({corr_val:+.1f}% change). "
            "Keystrokes were executed cleanly with minimal backspace repositioning."
        )
        c_badge = "Consistent"
        c_status = "stable"

    cards.append({
        "dimension": "Correction Behavior",
        "icon": "↻",
        "badge": c_badge,
        "status": c_status,
        "narrative": corr_text,
    })

    # 4. Rhythm Variability
    rhythm_val = indicators.get("rhythm_delta_pct", 0.0)
    sig_shifts = sum(1 for r in diag_results if r.get("rhythm_shift_significant") is True)
    if rhythm_val > 15.0:
        sig_note = f" (Mann-Whitney U confirmed significant rhythm distribution shifts in {sig_shifts} passage(s))" if sig_shifts > 0 else ""
        rhythm_text = (
            f"Keystroke rhythm variability was {abs(rhythm_val):.1f}% higher than during baseline calibration{sig_note}. "
            "Rather than maintaining a metronomic cadence, the timing between consecutive keystrokes varied dynamically."
        )
        r_badge = f"↑ {abs(rhythm_val):.1f}% Variable"
        r_status = "variable"
    else:
        rhythm_text = (
            f"Your keystroke rhythm remained steady across tasks (variability within {abs(rhythm_val):.1f}% of baseline). "
            "Typing cadence maintained high temporal regularity."
        )
        r_badge = "Uniform Cadence"
        r_status = "stable"

    cards.append({
        "dimension": "Rhythm Uniformity",
        "icon": "〰",
        "badge": r_badge,
        "status": r_status,
        "narrative": rhythm_text,
    })

    # 5. Task Consistency
    cons_val = indicators.get("consistency_score", 85.0)
    if cons_val >= 75.0:
        cons_text = (
            f"High cross-passage consistency ({cons_val:.0f}/100). "
            "Behavioral characteristics tracked symmetrically between the neutral recovery checks and the cognitive load passages."
        )
        cons_badge = f"{cons_val:.0f}/100 High"
        cons_status = "high"
    elif cons_val >= 50.0:
        cons_text = (
            f"Moderate cross-passage consistency ({cons_val:.0f}/100). "
            "Measurable behavioral shifts emerged primarily during the phonetic and sequencing load conditions."
        )
        cons_badge = f"{cons_val:.0f}/100 Moderate"
        cons_status = "moderate"
    else:
        cons_text = (
            f"Variable cross-passage consistency ({cons_val:.0f}/100). "
            "Typing dynamics fluctuated considerably across passages, suggesting external pacing shifts or task adaptation."
        )
        cons_badge = f"{cons_val:.0f}/100 Low"
        cons_status = "low"

    cards.append({
        "dimension": "Cross-Passage Consistency",
        "icon": "✓",
        "badge": cons_badge,
        "status": cons_status,
        "narrative": cons_text,
    })

    return cards


def interpret_behavioral_strain(
    strain_index: float,
    reliability_tier: str,
    indicators: Dict[str, Any],
    diag_results: List[Dict] = None,
    rel_data: Dict[str, Any] = None,
    baseline_stats: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Multi-Signal Behavioral Interpretation Engine.
    
    Synthesizes multiple observable signals into one of 5 user-facing behavioral states:
      1. STABLE TYPING PATTERN
      2. POSSIBLE INCREASED COGNITIVE STRAIN
      3. POSSIBLE REDUCED FOCUS / DISTRACTION
      4. POSSIBLE FATIGUE-LIKE PATTERN
      5. INCONCLUSIVE SESSION
    """
    if diag_results is None:
        diag_results = []
    if rel_data is None:
        rel_data = {}
    if baseline_stats is None:
        baseline_stats = {}

    spd_delta = float(indicators.get("speed_delta_pct", 0.0))
    pause_delta = float(indicators.get("pause_delta_pct", 0.0))
    rhythm_delta = float(indicators.get("rhythm_delta_pct", 0.0))
    corr_delta = float(indicators.get("correction_delta_pct", 0.0))
    cons_score = float(indicators.get("consistency_score", 85.0))

    rel_score = float(rel_data.get("reliability_score", 85.0))
    quality_score = float(rel_data.get("session_quality_score", 90.0))
    is_conclusive = bool(rel_data.get("is_conclusive", True))

    # Evidence audit tallies
    total_spikes = sum(int(r.get("hesitation_spikes", 0)) for r in diag_results)
    total_bursts = sum(int(r.get("correction_bursts", 0)) for r in diag_results)
    total_blurs = sum(int(r.get("blur_count", 0)) for r in diag_results)
    total_visibility_losses = sum(int(r.get("visibility_changes", 0)) for r in diag_results)
    total_pastes = sum(int(r.get("paste_attempts", 0)) for r in diag_results)
    total_unfocused_ms = sum(float(r.get("total_unfocused_ms", 0.0)) for r in diag_results)

    # Check temporal drift vs task-condition driven load
    task_load_driven = False
    monotonic_fatigue = False
    if len(diag_results) >= 4:
        p_speeds = [float(r.get("inter_key_mean_ms", r.get("avg_typing_speed_ms", 200.0))) for r in diag_results]
        neutral_avg_strain = (float(diag_results[0].get("strain_score", 0)) + float(diag_results[2].get("strain_score", 0))) / 2.0
        load_avg_strain = (float(diag_results[1].get("strain_score", 0)) + float(diag_results[3].get("strain_score", 0))) / 2.0
        
        # If Load tasks surged and Neutral tasks recovered:
        if (load_avg_strain - neutral_avg_strain) >= 20.0 or (p_speeds[1] > p_speeds[0] * 1.10 and p_speeds[2] < p_speeds[1] * 0.95):
            task_load_driven = True
        elif p_speeds[-1] > p_speeds[0] * 1.12 and p_speeds[2] >= p_speeds[1] * 0.92:
            monotonic_fatigue = True
    elif len(diag_results) >= 3:
        p_speeds = [float(r.get("inter_key_mean_ms", r.get("avg_typing_speed_ms", 200.0))) for r in diag_results]
        if p_speeds[-1] > p_speeds[0] * 1.15:
            monotonic_fatigue = True

    # Multi-Signal Pattern Decision Engine
    contributing_signals = []

    # Scenario 1: Inconclusive Check
    if not is_conclusive or rel_score < 40.0 or quality_score < 45.0:
        pattern_key = "INCONCLUSIVE"
        pattern_title = "Inconclusive Session"
        tier_label = "Inconclusive Session"
        one_sentence_summary = (
            "The observed signals did not agree strongly enough or continuity was insufficient to support a reliable behavioral interpretation."
        )
        detailed_explanation = (
            "This assessment contained data irregularities, paste attempts, or interruptions that reduced measurement reliability. "
            "Because keystroke timing must be captured in an unbroken, focused state to be meaningful, strong conclusions cannot be drawn. "
            "A repeat assessment under quiet, uninterrupted conditions is recommended."
        )
        contributing_signals.append("Measurement reliability or focus continuity fell below research thresholds.")

    # Scenario 2: Distraction / Reduced Focus Check
    elif (total_blurs >= 2 or total_visibility_losses >= 2 or total_pastes > 0 or total_unfocused_ms > 3500) and quality_score < 75.0:
        pattern_key = "REDUCED_FOCUS"
        pattern_title = "Possible Reduced Focus or Distraction"
        tier_label = "Possible Reduced Focus / Distraction"
        one_sentence_summary = (
            "Typing became more interrupted and irregular than your baseline, which may be consistent with reduced focus or environmental distraction during this session."
        )
        detailed_explanation = (
            f"Your session contained measurable focus interruptions ({total_blurs} window blur event(s) and {total_visibility_losses} tab switch(es)), "
            "accompanied by irregular typing rhythm. The session appears to have been interrupted or distracted rather than reflecting a stable, "
            "continuous cognitive state. Consequently, typing cadence changes during this session should be interpreted with caution."
        )
        contributing_signals.append(f"Recorded {total_blurs} focus loss event(s) and {total_visibility_losses} tab switch(es).")
        if rhythm_delta > 10.0:
            contributing_signals.append(f"Typing rhythm was {rhythm_delta:+.1f}% more variable than baseline.")

    # Scenario 3: Fatigue-Like Pattern (Monotonic progressive slowdown across tasks without task recovery)
    elif monotonic_fatigue and not task_load_driven and spd_delta > 8.0 and quality_score >= 65.0:
        pattern_key = "FATIGUE_LIKE"
        pattern_title = "Possible Fatigue-Like Pattern"
        tier_label = "Possible Fatigue-Like Pattern"
        one_sentence_summary = (
            "Typing speed gradually slowed and timing variability increased over the session, which may be consistent with fatigue-like changes or cumulative task difficulty."
        )
        detailed_explanation = (
            f"Your typing cadence slowed progressively across passages (averaging {spd_delta:+.1f}% wider gaps) "
            f"while pause intervals rose by {pause_delta:+.1f}%. Because focus continuity was high ({quality_score:.0f}%), "
            "this pattern is consistent with cumulative fatigue, hand/muscle tiredness, or increasing mental effort over time."
        )
        contributing_signals.append(f"Pacing slowed progressively across tasks (averaging {spd_delta:+.1f}% slower).")
        contributing_signals.append(f"Pause intervals widened by {pause_delta:+.1f}%.")

    # Scenario 4: Possible Increased Cognitive Strain (Task-Demand Driven)
    elif (task_load_driven or strain_index >= 55.0 or (spd_delta > 10.0 and pause_delta > 15.0) or (total_spikes >= 3 and rhythm_delta > 12.0)) and quality_score >= 60.0:
        pattern_key = "INCREASED_COGNITIVE_STRAIN"
        pattern_title = "Possible Increased Cognitive Strain"
        tier_label = "Possible Increased Cognitive Strain"
        
        # Build combinatorial narrative
        components = []
        if spd_delta > 8.0:
            components.append(f"slower typing (↓ {abs(spd_delta):.1f}%)")
            contributing_signals.append(f"Typing cadence slowed by {abs(spd_delta):.1f}% relative to resting baseline.")
        if pause_delta > 15.0 or total_spikes >= 2:
            spk_str = f" with {total_spikes} hesitation spikes" if total_spikes > 0 else ""
            components.append(f"increased hesitation (↑ {abs(pause_delta):.1f}% pauses{spk_str})")
            contributing_signals.append(f"Pause frequency increased by {abs(pause_delta):.1f}%{spk_str}.")
        if rhythm_delta > 10.0:
            components.append(f"greater rhythm variability (↑ {abs(rhythm_delta):.1f}%)")
            contributing_signals.append(f"Keystroke cadence variability rose by {abs(rhythm_delta):.1f}%.")
        if corr_delta > 12.0 or total_bursts >= 1:
            bst_str = f" with {total_bursts} correction burst(s)" if total_bursts > 0 else ""
            components.append(f"increased correction activity (↑ {abs(corr_delta):.1f}%{bst_str})")
            contributing_signals.append(f"Backspace editing increased by {abs(corr_delta):.1f}%{bst_str}.")

        comp_str = ", ".join(components) if components else "slower typing and elevated hesitation"
        one_sentence_summary = (
            f"The combination of {comp_str} is consistent with elevated cognitive or task-related strain during this session."
        )
        detailed_explanation = (
            f"Several typing characteristics changed simultaneously in a direction consistent with increased cognitive or task-related strain. "
            f"Under continuous focus ({quality_score:.0f}% session quality), your keystrokes exhibited {comp_str}. "
            "This combination typically emerges when motor planning demands elevate and individuals actively pause to sequence complex phrases."
        )

    # Scenario 5: Stable Typing Pattern
    else:
        pattern_key = "STABLE"
        pattern_title = "Stable Typing Pattern"
        tier_label = "Stable Typing Pattern"
        one_sentence_summary = (
            "Your typing pattern remained close to your personal baseline with no strong behavioral deviations detected."
        )
        detailed_explanation = (
            "Your typing cadence, pause frequency, and rhythm regularity remained consistent with your resting calibration baseline. "
            f"Typing speed was within {abs(spd_delta):.1f}% of baseline, pauses changed by {pause_delta:+.1f}%, and rhythm maintained high regularity. "
            "The observed dynamics suggest comfortable, steady execution across both neutral and challenging task conditions."
        )
        contributing_signals.append("Cadence and inter-key gaps remained steady across tasks.")
        contributing_signals.append("Pause frequency and rhythm variability showed low deviation from baseline.")

    # Contextual alternative factors (customized per pattern)
    cautious_context = (
        "Typing behavior is an indirect behavioral signal that can be influenced by many factors besides cognitive or emotional strain, including:\n"
        "  • Physical fatigue, tiredness, or poor sleep\n"
        "  • Environmental distractions, background noise, or multitasking\n"
        "  • Unfamiliar text phrasing, specialized vocabulary, or motor complexity\n"
        "  • Keyboard hardware differences (switch travel, laptop vs. mechanical)\n"
        "  • Seating posture and hand/wrist positioning\n"
        "  • Time of day and caffeine or stimulant intake"
    )

    # Scientific limitations & safety guidelines
    limitations = (
        "• MindType analyzes typing behavior; it does NOT directly measure or diagnose mental-health conditions.\n"
        "• The software cannot detect depression, anxiety, ADHD, burnout, or any psychiatric disorder.\n"
        "• Machine learning and statistical libraries (such as scikit-learn, TensorFlow, and SciPy) extract temporal feature distributions; "
        "they do not possess clinical diagnostic capabilities.\n"
        "• These results represent an experimental behavioral estimate from a single assessment.\n"
        "• A single session cannot determine a person's overall cognitive or mental-health status."
    )

    next_step = (
        "If you are experiencing persistent changes in concentration, stress, sleep, mood, or everyday wellbeing, "
        "consider discussing your experiences with a qualified healthcare or mental-health professional."
    )

    return {
        "pattern_key": pattern_key,
        "pattern_title": pattern_title,
        "tier_label": tier_label,
        "one_sentence_summary": one_sentence_summary,
        "summary": one_sentence_summary,
        "detailed_explanation": detailed_explanation,
        "contributing_signals": contributing_signals,
        "cautious_context": cautious_context,
        "limitations": limitations,
        "next_step": next_step,
    }
