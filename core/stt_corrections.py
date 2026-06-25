from __future__ import annotations

"""
Post-STT correction map: fixes systematic mishearings from faster-whisper
before the transcript reaches the LLM or the UI.

These are word-level substitutions applied after transcription.  Add entries
whenever you notice Whisper consistently transcribing a word incorrectly.

Format: { "whisper_output": "correct_word" }
Matching is case-insensitive; replacement preserves the casing of the
correct_word value you provide.
"""

import re

# ---------------------------------------------------------------------------
# Add Whisper mishearings here.
# ---------------------------------------------------------------------------
CORRECTIONS: dict[str, str] = {
    # Names whisper gets wrong
    "Adesh":    "Adarsh",
    "Aadesh":   "Adarsh",
    "Adash":    "Adarsh",
    "Aadarsh":  "Adarsh",
    "Rahool":   "Rahul",
    "Raahul":   "Rahul",
    # Common phrase mishearings
    "gonna":    "going to",   # optional: normalise informal speech for LLM
}

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), correct)
    for wrong, correct in CORRECTIONS.items()
]


def correct(text: str) -> str:
    """Apply all corrections to a raw STT transcript."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def add(wrong: str, right: str) -> None:
    """Dynamically register a new correction at runtime."""
    CORRECTIONS[wrong] = right
    _PATTERNS.append(
        (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), right)
    )
