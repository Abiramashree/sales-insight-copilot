"""Numeric fact-check: every number an LLM writes must trace to a source value.

Used as the deterministic half of the insight-card critic (M4) and as the
"factual accuracy" metric in the eval harness (M5).

A number in the text is SUPPORTED if it matches some source number under a
reasonable rendering: the same value, a percentage (0.231 -> 23 / 23.1),
thousands/millions (183080 -> 183k / 0.18M), or rounding to the precision written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION = re.compile(r"\[f\d+(?:\s*,\s*f\d+)*\]")
_DATE = re.compile(r"\b\d{4}[-‑‒–]\d{2}[-‑‒–]\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|"
                   r"Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{4})?\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|"
                   r"Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|"
                   r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b|\b20(?:1[0-9]|2[0-9])\b(?!\s?%)",
                   re.IGNORECASE)
_NUM = re.compile(r"(?<![\w.])([+\-−–‑‒]?)\s?€?\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?"
                  r"(%|pp|k\b|K\b|M\b|m\b|million\b|thousand\b)?")


@dataclass
class FactCheck:
    numbers: list[str] = field(default_factory=list)
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return 1.0 if not self.numbers else len(self.supported) / len(self.numbers)

    @property
    def passed(self) -> bool:
        return not self.unsupported


_SPACES = re.compile(r"[\u00a0\u202f\u2009\u2007]")
_RANGE = re.compile(r"(\d%?)\s*[–—]\s*(?=[+\-]?€?\d)")


_CODE = re.compile(r"`[^`]*`")                                   # inline code / SQL
_YEAR_MONTH = re.compile(r"\b\d{4}[-‑‒–]\d{2}\b(?![-‑‒–]\d)")       # 2016-12
_CONF_LEVEL = re.compile(r"\b(?:90|95|99)\s?%\s?(?=CI\b|confidence|interval|prediction)",
                         re.IGNORECASE)                                # "95% CI" is a label


def normalize(text: str) -> str:
    """Unicode spaces -> ' '; '70.1 – 77.9%' (a range) -> '70.1 to 77.9%' so the dash
    isn't read as a minus sign; drop inline code, year-month dates and confidence-level
    labels, which are not claims (all found by the M5 agent eval)."""
    text = _SPACES.sub(" ", text)
    text = _CONF_LEVEL.sub(" ", _YEAR_MONTH.sub(" ", _CODE.sub(" ", text)))
    return _RANGE.sub(r"\1 to ", text)


def extract_numbers(text: str) -> list[tuple[str, float, str]]:
    """Returns (raw_text, magnitude, unit) with dates and citation tags removed."""
    text = _DATE.sub(" ", _CITATION.sub(" ", normalize(text)))
    out = []
    for m in _NUM.finditer(text):
        sign, digits, unit = m.group(1), m.group(2), (m.group(3) or "").lower()
        v = float(digits.replace(",", ""))
        if unit in ("k", "thousand"):
            v *= 1e3
        elif unit in ("m", "million"):
            v *= 1e6
        if sign in ("-", "−", "–", "‑", "‒"):
            v = -v
        out.append((m.group(0).strip(), v, unit))
    return out


def _decimals(raw: str) -> int:
    m = re.search(r"\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def _matches(raw: str, v: float, unit: str, src: float) -> bool:
    d = _decimals(raw)
    signed = raw[:1] in "+-−–‑‒"
    if unit in ("%", "pp"):
        candidates = [src * 100, src]  # fraction stored, or already in percent
        tol = 0.5 * 10 ** -d + 1e-9
    elif unit in ("k", "thousand", "m", "million"):
        candidates = [src]
        tol = (1e3 if unit in ("k", "thousand") else 1e6) * 0.5 * 10 ** -d + 1e-9
    elif float(src).is_integer() and d == 0 and "€" not in raw:
        # IDs and counts (store 1066, 111 stores) must match exactly: a relative
        # tolerance would let "Store 1067" pass for store 1066 (found by M5 eval)
        candidates, tol = [src], 1e-9
    else:
        candidates = [src, src * 100] if abs(v) <= 100 else [src]
        tol = max(0.5 * 10 ** -d, abs(src) * 0.005) + 1e-9
    for c in candidates:
        if signed:
            # an explicit sign must agree: "+23%" for a -23% fact is a direction error
            if (raw[:1] == "+") == (c > 0) and abs(abs(v) - abs(c)) <= tol:
                return True
        # unsigned ("down 23%"): direction is in words, compare magnitudes
        elif abs(abs(v) - abs(c)) <= tol:
            return True
    return False


SMALL_ORDINALS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}  # "2 promo cycles", "top 10" etc.


def check(text: str, sources: list[float], allow_small: bool = True) -> FactCheck:
    fc = FactCheck()
    for raw, v, unit in extract_numbers(text):
        fc.numbers.append(raw)
        ok = any(_matches(raw, v, unit, s) for s in sources)
        if not ok and allow_small and not unit and v in SMALL_ORDINALS:
            ok = True
        (fc.supported if ok else fc.unsupported).append(raw)
    return fc


def numbers_in(obj) -> list[float]:
    """All numeric values inside a JSON-like structure (incl. numbers inside strings)."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(v for _, v, _ in extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(numbers_in(v))
    return out
