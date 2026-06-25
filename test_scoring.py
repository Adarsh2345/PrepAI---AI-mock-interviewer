"""
Quick smoke-test for the evaluator.
Feeds 3 fake Q&A pairs (strong / mediocre / weak) directly to Evaluator
and prints the scores. No mic, no VAD, no TTS needed.

Usage:
    python test_scoring.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

PAIRS = [
    {
        "label": "STRONG answer (expect 7-9)",
        "question": "Tell me about a time you had to deal with a difficult stakeholder.",
        "answer": (
            "At Heuristic Labs I was leading a document-processing project. "
            "A client kept changing requirements mid-sprint, which was blocking the team. "
            "I set up a weekly alignment call, created a shared requirements doc they had to "
            "sign off on before each sprint, and escalated one blocker to my manager when the "
            "client missed three consecutive sign-offs. After two weeks the churn stopped and "
            "we shipped on time. The client rated the delivery 9/10 in the post-project survey."
        ),
    },
    {
        "label": "MEDIOCRE answer (expect 4-6)",
        "question": "Describe a time you made a mistake that had a real impact.",
        "answer": (
            "Yeah I made a mistake once where I pushed some code without testing it properly "
            "and it broke something in production. I fixed it pretty quickly and learned to "
            "test more carefully after that."
        ),
    },
    {
        "label": "WEAK answer (expect 1-3)",
        "question": "Walk me through a situation where your initial approach failed.",
        "answer": "I don't really remember a specific example right now.",
    },
]


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

    if not api_key or api_key == "your_gemini_api_key_here":
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    print(f"\nUsing model: {model}")
    print("=" * 60)

    from interviewer.evaluator import Evaluator
    evaluator = Evaluator(llm_api_key=api_key, model=model)

    for pair in PAIRS:
        print(f"\n[{pair['label']}]")
        print(f"  Q: {pair['question'][:80]}...")
        print(f"  A: {pair['answer'][:80]}...")

        result = await evaluator.evaluate(pair["question"], pair["answer"])

        print(f"\n  Scores:")
        for dim in ("communication", "structure", "content_depth", "relevance"):
            d = result.get(dim, {})
            print(f"    {dim:<16} {d.get('score', '?'):>3}/10  — {d.get('reason', '')}")
        print(f"    {'overall':<16} {result.get('overall', '?'):>3}/10")
        print(f"    summary: {result.get('summary', '')}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
