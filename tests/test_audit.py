"""Tests for audit logging and replay."""

import json
import tempfile
from pathlib import Path

import pytest

from agentguard.core.events import LLMCallEvent, ToolCallEvent
from agentguard.logging.audit import AuditLogger
from agentguard.logging.reader import AuditReader


class TestAuditLogger:
    def test_writes_events(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        logger = AuditLogger(log_path)

        event = LLMCallEvent(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tokens_in=10,
            tokens_out=20,
        )
        logger.log(event)
        logger.close()

        # Verify the file was written
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        # Verify it's valid JSON
        data = json.loads(lines[0])
        assert data["model"] == "gpt-4o"
        assert data["event_type"] == "llm_call"

    def test_appends_multiple_events(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        logger = AuditLogger(log_path)

        for i in range(5):
            logger.log(LLMCallEvent(model=f"model-{i}"))
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_context_manager(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        with AuditLogger(log_path) as logger:
            logger.log(LLMCallEvent(model="gpt-4o"))

        assert log_path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "subdir" / "deep" / "test.jsonl"
        logger = AuditLogger(log_path)
        logger.log(LLMCallEvent())
        logger.close()
        assert log_path.exists()


class TestAuditReader:
    def _write_events(self, path, events):
        logger = AuditLogger(path)
        for e in events:
            logger.log(e)
        logger.close()

    def test_read_events(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        self._write_events(log_path, [
            LLMCallEvent(model="gpt-4o", run_id="run1"),
            ToolCallEvent(tool_name="search", run_id="run1"),
            LLMCallEvent(model="gpt-4o", run_id="run1"),
        ])

        reader = AuditReader(log_path)
        assert reader.total_events == 3
        assert reader.total_runs == 1

    def test_get_run(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        self._write_events(log_path, [
            LLMCallEvent(model="gpt-4o", run_id="run1", tokens_in=100, tokens_out=50, cost_usd=0.001),
            ToolCallEvent(tool_name="search", run_id="run1"),
            LLMCallEvent(model="gpt-4o", run_id="run2", tokens_in=200, tokens_out=100, cost_usd=0.002),
        ])

        reader = AuditReader(log_path)
        run1 = reader.get_run("run1")
        assert len(run1.events) == 2
        assert run1.total_cost == 0.001

        run2 = reader.get_run("run2")
        assert len(run2.events) == 1

    def test_get_all_runs(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        self._write_events(log_path, [
            LLMCallEvent(run_id="r1"),
            LLMCallEvent(run_id="r2"),
            LLMCallEvent(run_id="r1"),
        ])

        reader = AuditReader(log_path)
        runs = reader.get_all_runs()
        assert len(runs) == 2

    def test_print_trace(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        self._write_events(log_path, [
            LLMCallEvent(
                run_id="run1",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                response_content="Hi there!",
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.001,
            ),
            ToolCallEvent(
                run_id="run1",
                tool_name="search",
                arguments={"query": "test"},
                result="found it",
            ),
        ])

        reader = AuditReader(log_path)
        run = reader.get_run("run1")
        output = run.print_trace()

        assert "AgentGuard Run Trace" in output
        assert "gpt-4o" in output
        assert "search" in output

    def test_empty_file(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.touch()
        reader = AuditReader(log_path)
        assert reader.total_events == 0

    def test_nonexistent_file(self, tmp_path):
        reader = AuditReader(tmp_path / "nope.jsonl")
        assert reader.total_events == 0

    def test_total_cost(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        self._write_events(log_path, [
            LLMCallEvent(cost_usd=0.01, run_id="r1"),
            LLMCallEvent(cost_usd=0.02, run_id="r1"),
        ])
        reader = AuditReader(log_path)
        assert abs(reader.total_cost - 0.03) < 0.001
