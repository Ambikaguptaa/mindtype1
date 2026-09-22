"""
paragraphs.py
--------------
Five structured passages, typed in order, designed to measure natural typing
behavior under alternating task conditions:
  Passage 1: Calibration / natural typing baseline
  Passage 2: Neutral typing task 1
  Passage 3: Moderate cognitive-load task 1 (motor complexity / phonetic alternation)
  Passage 4: Neutral typing task 2 (recovery check / return-to-baseline)
  Passage 5: Moderate cognitive-load task 2 (dual-task processing / mental sequencing)

None of the passages contain emotionally distressing content. Task difficulty and
structural complexity provide measurable behavioral signal without inducing personal distress.

All passages are plain sentences with no dashes, numbers, brackets, or unusual symbols,
ensuring the assessment strictly examines typing flow and words.
"""

PASSAGES = [
    {
        "id": 1,
        "type": "calibration",
        "title": "Natural Typing Baseline (Calibration)",
        "condition": "Baseline Calibration",
        "description": "Establish your natural, resting typing rhythm.",
        "rationale": (
            "A calm, familiar descriptive passage without time pressure. "
            "Passage 1 calibrates this session's personal baseline for typing speed, "
            "inter-key interval, dwell time, pause frequency, and rhythm consistency. "
            "All subsequent passages are evaluated relative to this individual reference."
        ),
        "text": (
            "The quiet library was a good place to read on a slow afternoon. "
            "Most students had already left for the day and only the soft "
            "hum of the lights remained in the empty hall."
        ),
    },
    {
        "id": 2,
        "type": "neutral",
        "title": "Neutral Typing Task 1",
        "condition": "Neutral Condition",
        "description": "Standard narrative typing with neutral cognitive demand.",
        "rationale": (
            "A standard, neutral descriptive passage with everyday vocabulary. "
            "This provides the first controlled neutral measurement following calibration, "
            "verifying typing stability under ordinary task conditions."
        ),
        "text": (
            "Morning sunlight filtered gently through the kitchen window as coffee "
            "brewed on the counter. Outside on the quiet street birds chirped among "
            "the tall oak trees while a gentle breeze rustled the green leaves."
        ),
    },
    {
        "id": 3,
        "type": "load",
        "title": "Cognitive Load Task 1 (Motor & Phonetic Complexity)",
        "condition": "Cognitive Load Condition",
        "description": "Increased motor planning and phonetic alternation.",
        "rationale": (
            "Alternating phonetic patterns and high-consonant clustering introduce "
            "mechanical and motor-planning complexity. Under cognitive load, individuals "
            "frequently exhibit micro-hesitations, rhythm irregularity, and correction bursts."
        ),
        "text": (
            "Six sleek swans swiftly swam past six thick sticks stuck in shallow "
            "shoals while three free fleas fled from three brave bees buzzing "
            "beside the breezy beach."
        ),
    },
    {
        "id": 4,
        "type": "neutral",
        "title": "Neutral Typing Task 2 (Recovery Check)",
        "condition": "Neutral Condition",
        "description": "Evaluate behavioral return-to-baseline following load.",
        "rationale": (
            "A return to an easy, flowing narrative passage. Comparing Passage 4 against "
            "Passage 2 reveals whether typing rhythm recovers promptly or exhibits "
            "carryover fatigue after the preceding challenging task."
        ),
        "text": (
            "A small wooden boat rested quietly near the calm edge of the water. "
            "Across the wide lake the distant blue hills were covered in soft "
            "mist as clouds drifted slowly across the morning sky."
        ),
    },
    {
        "id": 5,
        "type": "load",
        "title": "Cognitive Load Task 2 (Mental Sequencing & Pacing)",
        "condition": "Cognitive Load Condition",
        "description": "Mild pacing awareness and dual-task mental demand.",
        "rationale": (
            "Introduces mild pacing awareness paired with an active sequencing cue. "
            "Dual-task demand challenges working memory and self-monitoring, revealing "
            "whether typing consistency holds under divided attention."
        ),
        "text": (
            "Type this passage at a steady pace while keeping your concentration "
            "on the sequence of each word. Notice how your hands move across the "
            "keys and maintain your natural rhythm until the final sentence ends."
        ),
    },
]


def get_passage(index: int) -> dict:
    """Returns passage dict for 0-indexed position (0..4)."""
    if 0 <= index < len(PASSAGES):
        return PASSAGES[index]
    raise IndexError(f"Passage index {index} out of range (0..{len(PASSAGES)-1})")


def total_passages() -> int:
    return len(PASSAGES)
