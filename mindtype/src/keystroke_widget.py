"""
keystroke_widget.py
----------------------
Python-side wrapper around the static HTML/JS component in
keystroke_component/. Declares the component once and exposes a simple
function the dashboard calls per passage.
"""
import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "keystroke_component")
_component_func = components.declare_component("keystroke_capture", path=_COMPONENT_DIR)


def keystroke_capture(paragraph: str, title: str = "", condition: str = "neutral", key=None):
    """Renders the research-grade capture widget. Returns None until the user
    finishes typing and clicks Submit, at which point it returns:
        {
          "events": [{interKeyMs, dwellMs, isBackspace, isError}, ...],
          "total_time_ms": float,
          "char_count": int,
          "quality": {
            "paste_attempts": int,
            "blur_count": int,
            "visibility_changes": int,
            "total_unfocused_ms": float,
            "longest_pause_ms": float,
            "idle_interruptions": int,
            "interruption_count": int
          }
        }
    """
    return _component_func(
        paragraph=paragraph,
        title=title,
        condition=condition,
        key=key,
        default=None
    )
