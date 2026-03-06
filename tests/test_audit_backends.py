"""Comprehensive tests for the audit backend system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentguard.core.events import LLMCallEvent, ToolCallEvent
from agentguard.logging.audit import AuditLogger
from agentguard.logging.backends.base import AuditBackend, AuditEntry, RetentionPolicy
from agentguard.logging.backends.local import LocalFileBackend


# ============================================================================
# AuditEntry
# ============================================================================

class TestAuditEntry:
    def test_compute_hash(self):
        entry = AuditEntry(sequence=0, data='{"test": true}', previous_hash="")
        h = entry.compute_hash()
        assert len(h) == 64  # SHA-256 hex digest
        assert h == entry.compute_hash()  # deterministic

    def test_fill_hash(self):
        entry = AuditEntry(sequence=0, data='{"test": true}')
        entry.fill_hash()
        assert entry.hash != ""
        assert entry.verify()

    def test_verify_correct(self):
        entry = AuditEntry(sequence=0, data='{"test": true}', previous_hash="")
        entry.fill_hash()
        assert entry.verify()

    def test_verify_tampered(self):
        entry = AuditEntry(sequence=0, data='{"test": true}', previous_hash="")
        entry.fill_hash()
        entry.data = '{"test": false}'  # tamper!
        assert not entry.verify()

    def test_hash_chain(self):
        """Two entries should form a valid chain."""
        e1 = AuditEntry(sequence=0, data='{"event": 1}', previous_hash="")
        e1.fill_hash()

        e2 = AuditEntry(sequence=1, data='{"event": 2}', previous_hash=e1.hash)
        e2.fill_hash()

        assert e2.previous_hash == e1.hash
        assert e1.verify()
        assert e2.verify()

    def test_to_json_and_from_json(self):
        entry = AuditEntry(sequence=42, data='{"model": "gpt-4"}', previous_hash="abc123")
        entry.fill_hash()

        json_str = entry.to_json()
        restored = AuditEntry.from_json(json_str)

        assert restored.sequence == 42
        assert restored.data == '{"model": "gpt-4"}'
        assert restored.previous_hash == "abc123"
        assert restored.hash == entry.hash
        assert restored.verify()

    def test_to_dict_and_from_dict(self):
        entry = AuditEntry(sequence=1, data='{"x": 1}')
        entry.fill_hash()
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.sequence == entry.sequence
        assert restored.hash == entry.hash


# ============================================================================
# LocalFileBackend
# ============================================================================

class TestLocalFileBackend:
    def test_write_and_read(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        backend = LocalFileBackend(log_path)

        entry = AuditEntry(sequence=0, data='{"test": true}', previous_hash="")
        entry.fill_hash()
        backend.write(entry)
        backend.close()

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_hash_chaining(self, tmp_path):
        log_path = tmp_path / "chain.jsonl"
        backend = LocalFileBackend(log_path)

        for i in range(5):
            entry = AuditEntry(sequence=i, data=json.dumps({"event": i}))
            backend.write(entry)

        backend.close()

        # Verify chain
        result = backend.verify_chain()
        assert result["valid"]
        assert result["total_entries"] == 5

    def test_verify_detects_tampering(self, tmp_path):
        log_path = tmp_path / "tamper.jsonl"
        backend = LocalFileBackend(log_path)

        for i in range(3):
            entry = AuditEntry(sequence=i, data=json.dumps({"event": i}))
            backend.write(entry)
        backend.close()

        # Tamper with middle line
        lines = log_path.read_text().strip().split("\n")
        data = json.loads(lines[1])
        data["data"] = '{"event": "tampered"}'
        lines[1] = json.dumps(data)
        log_path.write_text("\n".join(lines) + "\n")

        result = backend.verify_chain()
        assert not result["valid"]
        assert len(result["errors"]) > 0

    def test_read_with_filters(self, tmp_path):
        log_path = tmp_path / "filter.jsonl"
        backend = LocalFileBackend(log_path)

        for i in range(3):
            data = json.dumps({"event_type": "llm_call", "run_id": f"run_{i % 2}"})
            entry = AuditEntry(sequence=i, data=data)
            backend.write(entry)
        backend.close()

        entries = backend.read(run_id="run_0")
        assert len(entries) == 2

        entries = backend.read(run_id="run_1")
        assert len(entries) == 1

    def test_recovery_from_existing_file(self, tmp_path):
        log_path = tmp_path / "recover.jsonl"

        # Write some entries
        backend1 = LocalFileBackend(log_path)
        for i in range(3):
            entry = AuditEntry(sequence=i, data=json.dumps({"event": i}))
            backend1.write(entry)
        backend1.close()

        # Open a new backend on the same file
        backend2 = LocalFileBackend(log_path)
        assert backend2._sequence == 3
        assert backend2._last_hash != ""

        # Write more entries
        entry = AuditEntry(sequence=3, data=json.dumps({"event": 3}))
        backend2.write(entry)
        backend2.close()

        # Verify entire chain
        result = backend2.verify_chain()
        assert result["valid"]
        assert result["total_entries"] == 4

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "deep" / "nested" / "dir" / "test.jsonl"
        backend = LocalFileBackend(log_path)
        entry = AuditEntry(sequence=0, data='{"test": true}')
        backend.write(entry)
        backend.close()
        assert log_path.exists()

    def test_context_manager(self, tmp_path):
        log_path = tmp_path / "ctx.jsonl"
        with LocalFileBackend(log_path) as backend:
            entry = AuditEntry(sequence=0, data='{"test": true}')
            backend.write(entry)
        assert log_path.exists()

    def test_empty_file_verification(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.touch()
        backend = LocalFileBackend(log_path)
        result = backend.verify_chain()
        assert result["valid"]
        assert result["total_entries"] == 0

    def test_nonexistent_file_verification(self, tmp_path):
        backend = LocalFileBackend(tmp_path / "nope.jsonl")
        result = backend.verify_chain()
        assert result["valid"]


class TestLocalFileBackendEncryption:
    """Tests for encryption (requires cryptography package)."""

    @pytest.fixture
    def fernet_key(self):
        try:
            from cryptography.fernet import Fernet
            return Fernet.generate_key().decode()
        except ImportError:
            pytest.skip("cryptography package not installed")

    def test_encrypted_write_and_read(self, tmp_path, fernet_key):
        log_path = tmp_path / "encrypted.jsonl"
        backend = LocalFileBackend(log_path, encryption_key=fernet_key)

        original_data = '{"model": "gpt-4", "secret": "password123"}'
        entry = AuditEntry(sequence=0, data=original_data)
        backend.write(entry)
        backend.close()

        # Raw file should NOT contain the plaintext
        raw = log_path.read_text()
        assert "password123" not in raw

        # Read back should work (decrypts automatically)
        entries = backend.read()
        assert len(entries) == 1
        # The entry data is encrypted, need same key to decrypt
        from cryptography.fernet import Fernet
        f = Fernet(fernet_key.encode())
        decrypted = f.decrypt(entries[0].data.encode()).decode()
        assert "password123" in decrypted

    def test_encrypted_chain_verification(self, tmp_path, fernet_key):
        log_path = tmp_path / "enc_chain.jsonl"
        backend = LocalFileBackend(log_path, encryption_key=fernet_key)

        for i in range(5):
            entry = AuditEntry(sequence=i, data=json.dumps({"event": i}))
            backend.write(entry)
        backend.close()

        result = backend.verify_chain()
        assert result["valid"]
        assert result["total_entries"] == 5


# ============================================================================
# S3Backend (mocked)
# ============================================================================

class TestS3BackendMocked:
    def test_write_and_flush(self):
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            import importlib
            from agentguard.logging.backends import s3 as s3_module
            importlib.reload(s3_module)

            mock_client = MagicMock()
            with patch("boto3.client", return_value=mock_client):
                backend = s3_module.S3Backend(
                    bucket="test-bucket",
                    prefix="test",
                    buffer_size=2,
                )
                backend._s3 = mock_client

                entry1 = AuditEntry(sequence=0, data='{"test": 1}')
                entry1.fill_hash()
                entry2 = AuditEntry(sequence=1, data='{"test": 2}', previous_hash=entry1.hash)
                entry2.fill_hash()

                backend.write(entry1)
                # Should not flush yet (buffer_size=2)
                mock_client.put_object.assert_not_called()

                backend.write(entry2)
                # Should flush now
                mock_client.put_object.assert_called_once()

                # Verify put_object was called with correct bucket
                call_kwargs = mock_client.put_object.call_args[1]
                assert call_kwargs["Bucket"] == "test-bucket"
                assert call_kwargs["Key"].startswith("test/")


# ============================================================================
# SIEMBackend (mocked)
# ============================================================================

class TestSIEMBackendMocked:
    def test_splunk_format(self):
        with patch.dict("sys.modules", {"httpx": MagicMock()}):
            import importlib
            from agentguard.logging.backends import siem as siem_module
            importlib.reload(siem_module)

            backend = siem_module.SIEMBackend(
                provider="splunk",
                endpoint="https://splunk.test:8088/services/collector/event",
                token="test-token",
                buffer_size=100,
            )

            entry = AuditEntry(
                sequence=0,
                data='{"event_type": "llm_call", "model": "gpt-4"}',
            )
            entry.fill_hash()

            formatted = backend._format_splunk(entry)
            assert "event" in formatted
            assert "source" in formatted
            assert formatted["source"] == "agentguard"

    def test_datadog_format(self):
        with patch.dict("sys.modules", {"httpx": MagicMock()}):
            import importlib
            from agentguard.logging.backends import siem as siem_module
            importlib.reload(siem_module)

            backend = siem_module.SIEMBackend(
                provider="datadog",
                endpoint="https://logs.datadoghq.com/api/v2/logs",
                token="test-key",
                buffer_size=100,
            )

            entry = AuditEntry(sequence=0, data='{"event_type": "llm_call"}')
            entry.fill_hash()

            formatted = backend._format_datadog(entry)
            assert formatted["ddsource"] == "agentguard"
            assert "agentguard" in formatted
            assert formatted["agentguard"]["sequence"] == 0

    def test_invalid_provider(self):
        with patch.dict("sys.modules", {"httpx": MagicMock()}):
            import importlib
            from agentguard.logging.backends import siem as siem_module
            importlib.reload(siem_module)

            with pytest.raises(ValueError, match="Unknown SIEM provider"):
                siem_module.SIEMBackend(
                    provider="unknown",
                    endpoint="https://test.com",
                    token="test",
                )


# ============================================================================
# AuditLogger with Backends
# ============================================================================

class TestAuditLoggerWithBackends:
    def test_default_local_backend(self, tmp_path):
        log_path = tmp_path / "default.jsonl"
        logger = AuditLogger(log_path)

        event = LLMCallEvent(model="gpt-4o", tokens_in=10, tokens_out=20)
        logger.log(event)
        logger.close()

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        # Verify it's a hash-chained entry
        data = json.loads(lines[0])
        assert "hash" in data
        assert "prev_hash" in data
        assert "seq" in data

    def test_multiple_events_chained(self, tmp_path):
        log_path = tmp_path / "chain.jsonl"
        logger = AuditLogger(log_path)

        for i in range(5):
            logger.log(LLMCallEvent(model=f"model-{i}"))
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5

        # Verify chain
        prev_hash = ""
        for line in lines:
            data = json.loads(line)
            assert data["prev_hash"] == prev_hash
            prev_hash = data["hash"]

    def test_log_dict(self, tmp_path):
        log_path = tmp_path / "dict.jsonl"
        logger = AuditLogger(log_path)
        logger.log_dict({"custom_event": "test", "value": 42})
        logger.close()

        assert log_path.exists()

    def test_verify_chain(self, tmp_path):
        log_path = tmp_path / "verify.jsonl"
        logger = AuditLogger(log_path)

        for i in range(10):
            logger.log(LLMCallEvent(model=f"model-{i}"))
        logger.close()

        results = logger.verify()
        # Should have at least one backend result
        assert len(results) >= 1
        for name, result in results.items():
            assert result["valid"]
            assert result["total_entries"] == 10

    def test_context_manager(self, tmp_path):
        log_path = tmp_path / "ctx.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log(LLMCallEvent(model="gpt-4o"))
        assert log_path.exists()

    def test_multi_backend_fanout(self, tmp_path):
        path1 = tmp_path / "backend1.jsonl"
        path2 = tmp_path / "backend2.jsonl"

        backend1 = LocalFileBackend(path1)
        backend2 = LocalFileBackend(path2)

        logger = AuditLogger(backends=[backend1, backend2])

        event = LLMCallEvent(model="gpt-4o")
        logger.log(event)
        logger.close()

        # Both files should have the entry
        assert path1.exists()
        assert path2.exists()
        lines1 = path1.read_text().strip().split("\n")
        lines2 = path2.read_text().strip().split("\n")
        assert len(lines1) == 1
        assert len(lines2) == 1

    def test_path_property_backward_compat(self, tmp_path):
        log_path = tmp_path / "compat.jsonl"
        logger = AuditLogger(log_path)
        assert logger.path == log_path
        logger.close()


# ============================================================================
# Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Ensure existing AuditLogger API works unchanged."""

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "sub" / "deep" / "test.jsonl"
        logger = AuditLogger(log_path)
        logger.log(LLMCallEvent())
        logger.close()
        assert log_path.exists()

    def test_existing_test_audit_compat(self, tmp_path):
        """Original test_audit.py test should still pass."""
        log_path = tmp_path / "compat.jsonl"
        logger = AuditLogger(log_path)

        event = LLMCallEvent(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tokens_in=10,
            tokens_out=20,
        )
        logger.log(event)
        logger.close()

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        # Parse the hash-chained entry and extract inner data
        entry_data = json.loads(lines[0])
        inner_data = json.loads(entry_data["data"])
        assert inner_data["model"] == "gpt-4o"
        assert inner_data["event_type"] == "llm_call"

    def test_appends_multiple_events(self, tmp_path):
        log_path = tmp_path / "multi.jsonl"
        logger = AuditLogger(log_path)
        for i in range(5):
            logger.log(LLMCallEvent(model=f"model-{i}"))
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5


# ============================================================================
# RetentionPolicy
# ============================================================================

class TestRetentionPolicy:
    def test_default_values(self):
        policy = RetentionPolicy()
        assert policy.max_age_days == 0
        assert policy.max_entries == 0
        assert policy.max_size_bytes == 0

    def test_custom_values(self):
        policy = RetentionPolicy(max_age_days=30, max_entries=10000, max_size_bytes=1_000_000)
        assert policy.max_age_days == 30
        assert policy.max_entries == 10000
