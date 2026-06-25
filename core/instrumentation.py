from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnLatency:
    """Timing record for a single conversation turn (all times: monotonic seconds)."""
    turn_id: int

    # Absolute wall times
    t_speech_start: Optional[float] = None     # VAD detected user speech
    t_endpoint: Optional[float] = None          # VAD fired SPEECH_END
    t_stt_final: Optional[float] = None         # STT committed final transcript
    t_llm_first_token: Optional[float] = None   # first token from LLM
    t_tts_first_audio: Optional[float] = None   # first audio chunk from TTS
    t_playback_start: Optional[float] = None    # first audio written to sounddevice
    t_playback_end: Optional[float] = None      # last audio written (or barge-in)

    barge_in: bool = False                      # was this turn interrupted?
    transcript: str = ""
    response_preview: str = ""                  # first 80 chars of LLM response

    # ── Derived deltas (ms) ───────────────────────────────────────────────────

    @property
    def endpoint_to_llm_first_ms(self) -> Optional[float]:
        if self.t_endpoint and self.t_llm_first_token:
            return (self.t_llm_first_token - self.t_endpoint) * 1000
        return None

    @property
    def llm_first_to_tts_first_ms(self) -> Optional[float]:
        if self.t_llm_first_token and self.t_tts_first_audio:
            return (self.t_tts_first_audio - self.t_llm_first_token) * 1000
        return None

    @property
    def tts_first_to_playback_ms(self) -> Optional[float]:
        if self.t_tts_first_audio and self.t_playback_start:
            return (self.t_playback_start - self.t_tts_first_audio) * 1000
        return None

    @property
    def total_response_latency_ms(self) -> Optional[float]:
        """Endpoint → first audio out (the number that matters most)."""
        if self.t_endpoint and self.t_playback_start:
            return (self.t_playback_start - self.t_endpoint) * 1000
        return None

    @property
    def stt_lag_ms(self) -> Optional[float]:
        """Time from endpoint detection to STT final (should be ~0 or negative
        when STT final arrives before/with endpoint)."""
        if self.t_endpoint and self.t_stt_final:
            return (self.t_stt_final - self.t_endpoint) * 1000
        return None

    def summary(self) -> str:
        lines = [
            f"Turn {self.turn_id}{'  [BARGE-IN]' if self.barge_in else ''}",
            f"  transcript      : {self.transcript!r}",
            f"  stt_lag         : {self.stt_lag_ms:.1f} ms" if self.stt_lag_ms is not None else "  stt_lag         : —",
            f"  endpoint→LLM₁   : {self.endpoint_to_llm_first_ms:.1f} ms" if self.endpoint_to_llm_first_ms is not None else "  endpoint→LLM₁   : —",
            f"  LLM₁→TTS₁      : {self.llm_first_to_tts_first_ms:.1f} ms" if self.llm_first_to_tts_first_ms is not None else "  LLM₁→TTS₁      : —",
            f"  TTS₁→playback   : {self.tts_first_to_playback_ms:.1f} ms" if self.tts_first_to_playback_ms is not None else "  TTS₁→playback   : —",
            f"  ── TOTAL ──────: {self.total_response_latency_ms:.1f} ms" if self.total_response_latency_ms is not None else "  ── TOTAL ──────: —",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["endpoint_to_llm_first_ms"] = self.endpoint_to_llm_first_ms
        d["llm_first_to_tts_first_ms"] = self.llm_first_to_tts_first_ms
        d["tts_first_to_playback_ms"] = self.tts_first_to_playback_ms
        d["total_response_latency_ms"] = self.total_response_latency_ms
        d["stt_lag_ms"] = self.stt_lag_ms
        return d


class LatencyLogger:
    """Logs per-turn TurnLatency records to a JSONL file and stdout."""

    def __init__(self, log_path: str = "./logs/latency.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._turn_counter = 0

    def new_turn(self) -> TurnLatency:
        self._turn_counter += 1
        return TurnLatency(turn_id=self._turn_counter)

    def commit(self, record: TurnLatency) -> None:
        logger.info("\n%s", record.summary())
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")
