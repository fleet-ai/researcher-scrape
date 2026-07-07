"""Shared OpenAlex throttle across all callers.

Whether we're bulk-crawling with `oa_client.OAClient` or doing scattered
enrichment lookups via `build_outreach_list._openalex_get`, the requests
share one IP and one OpenAlex rate budget. If each caller throttles to
5 RPS locally, the aggregate is 10 RPS — which is what triggers rate
shaping. This module is the single choke point: everyone calls
`wait()` before every request, and `penalize()` after any 429.

Tuning:
- `OA_MIN_DELAY_S` (default 0.5s = 2 RPS) — sustained rate ceiling.
  OpenAlex docs say 10 RPS on the polite pool, but community reports peg
  the safe sustained rate at 2-3 RPS.
- `OA_COOLDOWN_S` (default 900s = 15 min) — after a 429, don't hit
  OpenAlex again for this long. OpenAlex 429s don't include Retry-After;
  the actual "you're throttled" window is 15+ min in practice, not the
  30-300s an exponential backoff assumes.
"""

import os
import threading
import time


_lock = threading.Lock()
_last_call = 0.0
_penalty_until = 0.0

_MIN_DELAY = float(os.environ.get("OA_MIN_DELAY_S", "0.5"))
_COOLDOWN = float(os.environ.get("OA_COOLDOWN_S", "900"))


def wait() -> None:
    """Block until it's OK to hit OpenAlex again.

    Applies both the sustained-rate delay and any active penalty window.
    """
    global _last_call
    with _lock:
        now = time.time()
        if now < _penalty_until:
            time.sleep(_penalty_until - now)
            now = time.time()
        gap = now - _last_call
        if gap < _MIN_DELAY:
            time.sleep(_MIN_DELAY - gap)
        _last_call = time.time()


def penalize(reason: str = "429") -> None:
    """Record a rate-limit hit. Extends the throttle window by COOLDOWN.

    Called by API clients when they get a 429. Subsequent `wait()` calls
    will block until the penalty expires.
    """
    global _penalty_until
    with _lock:
        _penalty_until = max(_penalty_until, time.time() + _COOLDOWN)


def penalty_remaining() -> float:
    """Seconds until any active penalty expires. 0 if none."""
    with _lock:
        return max(0.0, _penalty_until - time.time())
