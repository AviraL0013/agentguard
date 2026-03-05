"""Tests for the CLI audit reader."""

import json
import tempfile
from pathlib import Path

import pytest

from agentguard.cli import main as cli_main
from agentguard.core.events import LLMCallEvent, PolicyViolationEvent, ToolCallEvent
from agentguard.logging.audit import AuditLogger


def _create_test_audit(path: Path) -> None:
    """Create a realistic test audit log."""
    logger = AuditLogger(path)

    # Run 1: successful LLM call + tool call
    logger.log(LLMCallEvent(
        run_id="run_aaa111",
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the weather?"}],
        response_content="It's sunny today!",
        tokens_in=15,
        tokens_out=10,
        cost_usd=0.0001,
    ))
    logger.log(ToolCallEvent(
        run_id="run_aaa111",
        tool_name="get_weather",
        arguments={"city": "Mumbai"},
        result="Sunny, 32C",
        duration_ms=120,
    ))
    logger.log(LLMCallEvent(
        run_id="run_aaa111",
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize the weather"}],
        response_content="Mumbai is sunny at 32C",
        tokens_in=20,
        tokens_out=15,
        cost_usd=0.0002,
    ))

    # Run 2: blocked call with violation
    logger.log(LLMCallEvent(
        run_id="run_bbb222",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
        blocked=True,
        block_reason="PII detected: ssn",
        input_pii_found=[{"type": "ssn", "value": "123-45-6789"}],
        policy_results=[{"policy": "pii", "action": "block", "reason": "PII detected"}],
    ))
    logger.log(PolicyViolationEvent(
        run_id="run_bbb222",
        policy_name="pii",
        violation_reason="SSN detected in input",
        action_taken="block",
    ))

    # Run 3: tool blocked
    logger.log(ToolCallEvent(
        run_id="run_ccc333",
        tool_name="delete_database",
        arguments={"db": "production"},
        blocked=True,
        block_reason="Tool 'delete_database' is blocked by policy",
    ))

    logger.close()


class TestCLIRuns:
    def test_runs_command(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "runs"])
        captured = capsys.readouterr()
        assert "run_aaa111" in captured.out
        assert "run_bbb222" in captured.out

    def test_runs_empty(self, tmp_path, capsys):
        log = tmp_path / "empty.jsonl"
        log.touch()
        cli_main(["--file", str(log), "runs"])
        captured = capsys.readouterr()
        assert "No runs" in captured.out


class TestCLIReplay:
    def test_replay_run(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "replay", "run_aaa111"])
        captured = capsys.readouterr()
        assert "run_aaa111" in captured.out
        assert "gpt-4o" in captured.out
        assert "get_weather" in captured.out

    def test_replay_blocked_run(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "replay", "run_bbb222"])
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.out
        assert "PII" in captured.out or "pii" in captured.out

    def test_replay_nonexistent(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "replay", "nonexistent"])
        captured = capsys.readouterr()
        assert "No events" in captured.out


class TestCLIViolations:
    def test_violations_found(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "violations"])
        captured = capsys.readouterr()
        assert "pii" in captured.out.lower()
        assert "VIOLATIONS" in captured.out or "violation" in captured.out.lower()

    def test_no_violations(self, tmp_path, capsys):
        log = tmp_path / "clean.jsonl"
        logger = AuditLogger(log)
        logger.log(LLMCallEvent(model="gpt-4o", tokens_in=10, tokens_out=5))
        logger.close()
        cli_main(["--file", str(log), "violations"])
        captured = capsys.readouterr()
        assert "No policy violations" in captured.out


class TestCLIStats:
    def test_stats_dashboard(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "stats"])
        captured = capsys.readouterr()
        assert "DASHBOARD" in captured.out
        assert "gpt-4o" in captured.out
        assert "get_weather" in captured.out


class TestCLISearch:
    def test_search_model(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "search", "gpt-4o-mini"])
        captured = capsys.readouterr()
        assert "1 match" in captured.out or "matches" in captured.out

    def test_search_no_results(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "search", "nonexistent_xyz"])
        captured = capsys.readouterr()
        assert "No events" in captured.out


class TestCLIExport:
    def test_export_json_to_file(self, tmp_path):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        output = tmp_path / "export.json"
        cli_main(["--file", str(log), "export", "--format", "json", "-o", str(output)])
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) > 0

    def test_export_csv_to_file(self, tmp_path):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        output = tmp_path / "export.csv"
        cli_main(["--file", str(log), "export", "--format", "csv", "-o", str(output)])
        assert output.exists()
        content = output.read_text()
        assert "event_type" in content
        assert "llm_call" in content

    def test_export_filtered_by_run(self, tmp_path):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        output = tmp_path / "export.json"
        cli_main(["--file", str(log), "export", "--format", "json", "-o", str(output), "--run-id", "run_aaa111"])
        data = json.loads(output.read_text())
        assert all(e.get("run_id") == "run_aaa111" for e in data)

    def test_export_json_stdout(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        _create_test_audit(log)
        cli_main(["--file", str(log), "export", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) > 0


class TestCLIHelp:
    def test_no_command_shows_help(self, tmp_path, capsys):
        cli_main(["--file", str(tmp_path / "test.jsonl")])
        captured = capsys.readouterr()
        assert "agentguard" in captured.out.lower() or "usage" in captured.out.lower()
