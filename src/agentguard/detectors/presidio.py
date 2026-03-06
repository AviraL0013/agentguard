"""Production-grade PII detection using Microsoft Presidio + spaCy NER.

Requires optional dependencies::

    pip install agentaudit-sdk[presidio]

Or manually::

    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

from typing import Optional

from agentguard.detectors.pii import PIIMatch


# ---------------------------------------------------------------------------
# Entity type mapping: Presidio names → AgentGuard conventions
# ---------------------------------------------------------------------------

_ENTITY_MAP: dict[str, str] = {
    "PERSON": "person",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "CREDIT_CARD": "credit_card",
    "IP_ADDRESS": "ip_address",
    "DATE_TIME": "date_of_birth",
    "LOCATION": "location",
    "NRP": "nationality",  # Nationality, Religious, or Political group
    "MEDICAL_LICENSE": "medical_license",
    "US_BANK_NUMBER": "bank_account",
    "US_DRIVER_LICENSE": "driver_license",
    "US_ITIN": "itin",
    "US_PASSPORT": "passport",
    "UK_NHS": "nhs_number",
    "IBAN_CODE": "iban",
    "CRYPTO": "crypto_address",
}


class PresidioPIIDetector:
    """Production-grade PII detection backed by Microsoft Presidio.

    Uses spaCy NER models for entity recognition with support for 50+
    entity types, confidence scoring, and multi-language detection.

    Usage::

        from agentguard.detectors.presidio import PresidioPIIDetector

        detector = PresidioPIIDetector(score_threshold=0.5)
        matches = detector.scan("Contact John Smith at john@example.com")

    Args:
        languages: Languages to detect PII in (default: ``["en"]``).
        score_threshold: Minimum confidence score (0.0–1.0) to include
            a match. Lower values catch more but increase false positives.
        entities: Specific Presidio entity types to detect. ``None`` means
            detect all available entity types.
        deny_list: Extra terms to always flag as PII (custom blocklist).
    """

    def __init__(
        self,
        *,
        languages: list[str] | None = None,
        score_threshold: float = 0.5,
        entities: list[str] | None = None,
        deny_list: list[str] | None = None,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:
            raise ImportError(
                "Presidio is not installed. Install it with:\n"
                "  pip install agentaudit-sdk[presidio]\n"
                "Then download a spaCy model:\n"
                "  python -m spacy download en_core_web_lg"
            ) from exc

        self._languages = languages or ["en"]
        self._threshold = score_threshold
        self._entities = entities or [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
            "IP_ADDRESS", "LOCATION", "NRP", "MEDICAL_LICENSE", "US_BANK_NUMBER",
            "US_DRIVER_LICENSE", "US_PASSPORT", "UK_NHS", "IBAN_CODE", "CRYPTO"
        ]
        self._deny_list = deny_list

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    # -- public API (PIIDetector protocol) ---------------------------------

    def scan(self, text: str) -> list[PIIMatch]:
        """Scan *text* and return all PII matches found."""
        if not text:
            return []

        ad_hoc_recognizers = []
        if self._deny_list:
            from presidio_analyzer import PatternRecognizer
            ad_hoc_recognizers.append(
                PatternRecognizer(supported_entity="CUSTOM_DENY", deny_list=self._deny_list)
            )

        results = self._analyzer.analyze(
            text=text,
            language=self._languages[0],
            entities=self._entities,
            score_threshold=self._threshold,
            ad_hoc_recognizers=ad_hoc_recognizers if ad_hoc_recognizers else None,
        )

        matches: list[PIIMatch] = []
        for r in results:
            pii_type = _ENTITY_MAP.get(r.entity_type, r.entity_type.lower())
            matches.append(
                PIIMatch(
                    pii_type=pii_type,
                    value=text[r.start : r.end],
                    start=r.start,
                    end=r.end,
                    confidence=r.score,
                )
            )

        # Sort by position for consistent output
        matches.sort(key=lambda m: m.start)
        return matches

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
