"""Tests for the Presidio-based PII detector.

All tests are skipped if presidio-analyzer is not installed.
Install with: pip install agentaudit-sdk[presidio]
Then: python -m spacy download en_core_web_lg
"""

import pytest

presidio_analyzer = pytest.importorskip("presidio_analyzer", reason="presidio-analyzer not installed")

from agentguard.detectors.presidio import PresidioPIIDetector
from agentguard.detectors.pii import PIIDetector, PIIMatch


@pytest.fixture
def detector():
    """Create a PresidioPIIDetector with default settings."""
    return PresidioPIIDetector(score_threshold=0.4)


class TestPresidioPIIDetector:
    """Core detection capabilities."""

    def test_detect_person_name(self, detector):
        """The primary capability gap — regex can't do this."""
        matches = detector.scan("Please contact John Smith for more details")
        person_matches = [m for m in matches if m.pii_type == "person"]
        assert len(person_matches) >= 1
        assert any("John" in m.value for m in person_matches)

    def test_detect_email(self, detector):
        matches = detector.scan("Send to john@example.com please")
        email_matches = [m for m in matches if m.pii_type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "john@example.com"

    def test_detect_phone(self, detector):
        matches = detector.scan("Call me at 555-123-4567")
        phone_matches = [m for m in matches if m.pii_type == "phone"]
        assert len(phone_matches) >= 1

    def test_detect_ssn(self, detector):
        matches = detector.scan("My social security number is 012-34-5678")
        ssn_matches = [m for m in matches if m.pii_type == "ssn"]
        assert len(ssn_matches) >= 1

    def test_detect_credit_card(self, detector):
        matches = detector.scan("Card: 4111111111111111")
        cc_matches = [m for m in matches if m.pii_type == "credit_card"]
        assert len(cc_matches) >= 1

    def test_detect_multiple_entity_types(self, detector):
        text = "John Smith's email is john@example.com and phone is 555-123-4567"
        matches = detector.scan(text)
        types_found = {m.pii_type for m in matches}
        # Should detect at least person and email
        assert "person" in types_found
        assert "email" in types_found

    def test_no_pii_in_clean_text(self, detector):
        matches = detector.scan("What is the weather today in the city?")
        # Should be empty or only very low-confidence spurious matches
        high_confidence = [m for m in matches if m.confidence > 0.7]
        assert len(high_confidence) == 0


class TestConfidenceScoring:
    """Confidence score behavior."""

    def test_confidence_field_present(self, detector):
        matches = detector.scan("Contact John Smith at john@example.com")
        assert all(0.0 < m.confidence <= 1.0 for m in matches)

    def test_high_threshold_fewer_matches(self):
        low = PresidioPIIDetector(score_threshold=0.3)
        high = PresidioPIIDetector(score_threshold=0.9)
        text = "John Smith lives at 123 Main St, Springfield"
        low_matches = low.scan(text)
        high_matches = high.scan(text)
        assert len(high_matches) <= len(low_matches)


class TestRedaction:
    """Redaction capabilities."""

    def test_redact_replaces_pii(self, detector):
        text = "Contact John Smith at john@example.com"
        redacted = detector.redact(text)
        assert "john@example.com" not in redacted
        assert "[REDACTED_" in redacted

    def test_redact_empty_string(self, detector):
        assert detector.redact("") == ""

    def test_redact_no_pii(self, detector):
        text = "Hello world, no sensitive data"
        assert detector.redact(text) == text


class TestHasPII:
    """has_pii() quick check."""

    def test_has_pii_true(self, detector):
        assert detector.has_pii("Email: john@example.com")

    def test_has_pii_false(self, detector):
        assert not detector.has_pii("The quick brown fox jumps over the lazy dog")

    def test_empty_string(self, detector):
        assert not detector.has_pii("")


class TestEdgeCases:
    """Edge cases and configuration."""

    def test_empty_string_scan(self, detector):
        assert detector.scan("") == []

    def test_entity_filtering(self):
        detector = PresidioPIIDetector(entities=["EMAIL_ADDRESS"])
        text = "John Smith's email is john@example.com"
        matches = detector.scan(text)
        types_found = {m.pii_type for m in matches}
        assert "email" in types_found
        # Should NOT detect person names when only EMAIL_ADDRESS is requested
        assert "person" not in types_found

    def test_scan_returns_sorted_by_position(self, detector):
        text = "john@example.com belongs to John Smith"
        matches = detector.scan(text)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].start <= matches[i + 1].start


class TestProtocolCompliance:
    """Verify PresidioPIIDetector satisfies the PIIDetector protocol."""

    def test_isinstance_check(self, detector):
        assert isinstance(detector, PIIDetector)


class TestIntegration:
    """Integration with PIIPolicy and AgentGuard."""

    def test_with_pii_policy(self, detector):
        from agentguard.core.events import LLMCallEvent, PolicyAction
        from agentguard.policies.pii_policy import PIIPolicy

        policy = PIIPolicy(detector=detector)
        event = LLMCallEvent(
            messages=[{"role": "user", "content": "Contact John Smith for details"}]
        )
        result = policy.evaluate(event)
        # Should block because person name is PII
        assert result.blocked
        assert "person" in result.reason.lower()

    def test_with_agentguard(self, tmp_path, detector):
        from agentguard.core.guard import AgentGuard
        from agentguard.core.interceptor import PolicyViolationError

        guard = AgentGuard(
            policies=["pii"],
            audit_path=tmp_path / "test.jsonl",
            pii_detector=detector,
        )

        def greet(name: str) -> str:
            return f"Hello {name}"

        safe_greet = guard.wrap_tool(greet)

        # Should block because "John Smith" is a person name
        with pytest.raises(PolicyViolationError):
            safe_greet(name="John Smith")

        guard.close()
