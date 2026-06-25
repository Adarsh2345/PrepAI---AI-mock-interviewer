from __future__ import annotations

"""
Pronunciation normaliser: rewrites words that TTS mispronounces before text
reaches the synthesis engine.

Approach: regex word-boundary substitutions so "Rahul" → "Raahul", etc.
Add entries to SUBSTITUTIONS for any name or term the TTS gets wrong.
Keys are matched case-insensitively; replacement preserves the written form
shown in transcripts (only the TTS-bound text is rewritten).

Format: { "original": "spoken_spelling" }
"""

import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Add mispronounced words here.
# Use phonetic respelling that the TTS engine pronounces correctly:
#   - Stretch vowels with repeated letters (aa, ee, oo) for longer sounds
#   - Split consonant clusters with a schwa (uh) if needed
#   - Use hyphens to force syllable breaks where useful
# ---------------------------------------------------------------------------
SUBSTITUTIONS: dict[str, str] = {
    # Indian names
    "Rahul":    "Raahul",
    "Priya":    "Preeya",
    "Arjun":    "Aarjun",
    "Ananya":   "Aanaanya",
    "Vikram":   "Vikruhm",
    "Kavya":    "Kaavya",
    "Rohan":    "Rohan",      # usually fine, listed for completeness
    "Shreya":   "Shraya",
    "Aditya":   "Aaditya",
    "Pooja":    "Pooja",
    "Riya":     "Reea",
    "Aryan":    "Aaryan",
    "Neha":     "Nayha",
    "Kunal":    "Koonarl",
    "Divya":    "Divya",
    "Sanjay":   "Sunjay",
    "Meera":    "Meera",
    "Karan":    "Karuhn",
    "Ankit":    "Ankeet",
    "Deepak":   "Deepuk",
    # Common tech terms TTS often garbles
    "API":      "A P I",
    "LLM":      "L L M",
    "GPT":      "G P T",
    "UI":       "U I",
    "async":    "ay-sink",
}

# Pre-compile patterns: \b word boundaries, case-insensitive
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(original)}\b", re.IGNORECASE), replacement)
    for original, replacement in SUBSTITUTIONS.items()
]


def normalise(text: str) -> str:
    """Apply all substitutions to a block of text destined for TTS."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def add(original: str, spoken: str) -> None:
    """Dynamically register a new substitution at runtime."""
    SUBSTITUTIONS[original] = spoken
    _PATTERNS.append(
        (re.compile(rf"\b{re.escape(original)}\b", re.IGNORECASE), spoken)
    )
