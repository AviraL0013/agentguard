"""Regex-based PII detection and redaction.

Pluggable interface: swap RegexPIIDetector for Presidio or ML-based
detection later without changing calling code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    """A single PII detection result."""

    pii_type: str          # e.g. "email", "phone", "ssn"
    value: str             # the matched text
    start: int             # start position in original text
    end: int               # end position in original text
    redacted_label: str = ""  # e.g. "[REDACTED_EMAIL]"

    def __post_init__(self):
        if not self.redacted_label:
            self.redacted_label = f"[REDACTED_{self.pii_type.upper()}]"


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class PIIDetector(Protocol):
    """Interface for PII detection — implement this to provide custom detection."""

    def scan(self, text: str) -> list[PIIMatch]:
        """Scan text and return all PII matches."""
        ...

    def redact(self, text: str) -> str:
        """Redact all PII from text, replacing with labels."""
        ...


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: list[tuple[str, str, int]] = [
    # (pii_type, regex_pattern, regex_flags)
    (
        "email",
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        0,
    ),
    (
        "phone",
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)",
        0,
    ),
    (
        "ssn",
        r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
        0,
    ),
    (
        "credit_card",
        r"(?<!\d)(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)",
        0,
    ),
    (
        "ip_address",
        r"(?<!\d)(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}(?!\d)",
        0,
    ),
    (
        "date_of_birth",
        r"(?i)(?:dob|date of birth|born on|birthday)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Regex-based implementation
# ---------------------------------------------------------------------------

class RegexPIIDetector:
    """Regex-based PII detection — suitable for MVP.

    Supports built-in patterns for email, phone, SSN, credit cards,
    IP addresses, and dates of birth. Add custom patterns via
    ``add_pattern()``.
    """

    def __init__(self, *, enable_builtins: bool = True) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        if enable_builtins:
            for pii_type, pattern, flags in _BUILTIN_PATTERNS:
                self._patterns.append((pii_type, re.compile(pattern, flags)))

    # -- public API --------------------------------------------------------

    def add_pattern(self, pii_type: str, pattern: str, flags: int = 0) -> None:
        """Register a custom PII pattern."""
        self._patterns.append((pii_type, re.compile(pattern, flags)))

    def scan(self, text: str) -> list[PIIMatch]:
        """Scan *text* and return all PII matches found."""
        matches: list[PIIMatch] = []
        for pii_type, compiled in self._patterns:
            for m in compiled.finditer(text):
                # For patterns with groups, use the first group; else full match
                value = m.group(1) if m.lastindex else m.group(0)
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        start=m.start(1) if m.lastindex else m.start(),
                        end=m.end(1) if m.lastindex else m.end(),
                    )
                )
        # Sort by position and deduplicate overlapping matches
        matches.sort(key=lambda x: x.start)
        return self._deduplicate(matches)

    def redact(self, text: str) -> str:
        """Replace all detected PII in *text* with redaction labels."""
        matches = self.scan(text)
        if not matches:
            return text
        # Build result by replacing matches back-to-front to preserve positions
        result = list(text)
        for match in reversed(matches):
            result[match.start : match.end] = list(match.redacted_label)
        return "".join(result)

    def has_pii(self, text: str) -> bool:
        """Quick check — does *text* contain any PII?"""
        return len(self.scan(text)) > 0

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _deduplicate(matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping the longer one."""
        if not matches:
            return matches
        result: list[PIIMatch] = [matches[0]]
        for m in matches[1:]:
            prev = result[-1]
            if m.start >= prev.end:
                result.append(m)
            elif (m.end - m.start) > (prev.end - prev.start):
                result[-1] = m
        return result
