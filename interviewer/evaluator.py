from __future__ import annotations

"""
Per-answer evaluator: calls Gemini (non-streamed) after each answer to score
it across 4 dimensions. Runs in a thread executor so it doesn't block the
pipeline event loop. Uses google.genai (new SDK).
"""

import asyncio
import json
import logging
import re
from typing import Optional

from interviewer.prompts import eval_prompt

logger = logging.getLogger(__name__)

_FALLBACK = {
    "communication": {"score": 5, "reason": "Could not evaluate"},
    "structure":     {"score": 5, "reason": "Could not evaluate"},
    "content_depth": {"score": 5, "reason": "Could not evaluate"},
    "relevance":     {"score": 5, "reason": "Could not evaluate"},
    "overall": 5,
    "summary": "Evaluation unavailable for this answer.",
}

_REQUIRED_KEYS = {"communication", "structure", "content_depth", "relevance", "overall"}


def _extract_json(text: str) -> Optional[dict]:
    # Strip markdown code fences (``` or ```json ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try full parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block (non-greedy won't work with nested,
    # so use a brace-counting approach)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
                break

    logger.warning("Evaluator: JSON extraction failed. Raw text: %r", text[:400])
    return None


def _validate(data: dict) -> bool:
    """Return True only if the parsed dict has all required keys with numeric scores."""
    for key in _REQUIRED_KEYS:
        if key not in data:
            logger.warning("Evaluator: missing key %r in response", key)
            return False
    for dim in ("communication", "structure", "content_depth", "relevance"):
        val = data[dim]
        if not isinstance(val, dict) or "score" not in val:
            logger.warning("Evaluator: malformed dim %r: %r", dim, val)
            return False
        if not isinstance(val["score"], (int, float)):
            logger.warning("Evaluator: non-numeric score for %r: %r", dim, val["score"])
            return False
    if not isinstance(data["overall"], (int, float)):
        logger.warning("Evaluator: non-numeric overall: %r", data["overall"])
        return False
    return True


class Evaluator:
    def __init__(self, llm_api_key: str, model: str = "gemini-2.0-flash-lite") -> None:
        self._api_key = llm_api_key
        self._model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _evaluate_sync(self, question: str, answer: str) -> dict:
        if not answer or not answer.strip():
            logger.info("Evaluator: empty answer — returning fallback")
            return _FALLBACK
        try:
            from google.genai import types
            client = self._get_client()
            prompt = eval_prompt(question, answer)
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=600,
                ),
            )
            raw = response.text
            logger.info("Evaluator raw response (first 300 chars): %s", raw[:300])

            result = _extract_json(raw)
            if result is None:
                logger.error("Evaluator: could not extract JSON. Full response: %s", raw)
                return _FALLBACK

            if not _validate(result):
                logger.error("Evaluator: JSON structure invalid: %s", result)
                return _FALLBACK

            logger.info(
                "Evaluator: Q scored — comm=%s struct=%s depth=%s rel=%s overall=%s",
                result["communication"]["score"],
                result["structure"]["score"],
                result["content_depth"]["score"],
                result["relevance"]["score"],
                result["overall"],
            )
            return result

        except Exception as e:
            logger.error("Evaluator error: %s", e, exc_info=True)
            return _FALLBACK

    async def evaluate(self, question: str, answer: str) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._evaluate_sync, question, answer)
