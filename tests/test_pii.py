"""Tests for PII detection."""

import pytest
from agentguard.detectors.pii import RegexPIIDetector


@pytest.fixture
def detector():
    return RegexPIIDetector()


class TestPIIDetection:
    def test_detect_email(self, detector):
        matches = detector.scan("Contact me at john@example.com please")
        assert len(matches) == 1
        assert matches[0].pii_type == "email"
        assert matches[0].value == "john@example.com"

    def test_detect_multiple_emails(self, detector):
        matches = detector.scan("Email john@example.com or jane@test.org")
        assert len(matches) == 2
        assert all(m.pii_type == "email" for m in matches)

    def test_detect_phone(self, detector):
        matches = detector.scan("Call me at 555-123-4567")
        assert len(matches) >= 1
        assert any(m.pii_type == "phone" for m in matches)

    def test_detect_phone_with_area_code(self, detector):
        matches = detector.scan("Phone: (555) 123-4567")
        assert any(m.pii_type == "phone" for m in matches)

    def test_detect_ssn(self, detector):
        matches = detector.scan("SSN: 123-45-6789")
        assert len(matches) >= 1
        assert any(m.pii_type == "ssn" for m in matches)

    def test_detect_credit_card(self, detector):
        matches = detector.scan("Card: 4111-1111-1111-1111")
        assert any(m.pii_type == "credit_card" for m in matches)

    def test_detect_ip_address(self, detector):
        matches = detector.scan("Server at 192.168.1.100")
        assert any(m.pii_type == "ip_address" for m in matches)

    def test_no_pii(self, detector):
        matches = detector.scan("Hello world, nothing sensitive here")
        assert len(matches) == 0

    def test_has_pii_true(self, detector):
        assert detector.has_pii("Email: test@example.com")

    def test_has_pii_false(self, detector):
        assert not detector.has_pii("No sensitive data here")

    def test_redact_email(self, detector):
        result = detector.redact("Contact john@example.com")
        assert "john@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_redact_multiple(self, detector):
        text = "Email john@example.com, phone 555-123-4567"
        result = detector.redact(text)
        assert "john@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_custom_pattern(self, detector):
        detector.add_pattern("employee_id", r"EMP-\d{6}")
        matches = detector.scan("Employee EMP-123456 reported in")
        assert len(matches) == 1
        assert matches[0].pii_type == "employee_id"

    def test_empty_string(self, detector):
        assert detector.scan("") == []
        assert detector.redact("") == ""
