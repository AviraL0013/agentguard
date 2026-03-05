"""AgentGuard CLI — audit reader and forensics tool.

Run from the command line:
    agentguard runs                     # list all runs
    agentguard replay <run_id>          # step-by-step replay
    agentguard violations               # show all policy violations
    agentguard stats                    # dashboard overview
    agentguard search <query>           # search events
    agentguard export --format csv      # export to CSV/JSON
    agentguard tail                     # live tail (watch mode)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agentguard.logging.reader import AuditReader, RunTrace

# ---------------------------------------------------------------------------
# ANSI color helpers (no dependencies)
# ---------------------------------------------------------------------------

class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

C = _Colors()


def _c(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


def _header(text: str) -> str:
    return _c(text, C.BOLD + C.CYAN)


def _success(text: str) -> str:
    return _c(text, C.GREEN)


def _warn(text: str) -> str:
    return _c(text, C.YELLOW)


def _error(text: str) -> str:
    return _c(text, C.RED)


def _dim(text: str) -> str:
    return _c(text, C.DIM)


def _bold(text: str) -> str:
    return _c(text, C.BOLD)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _render_table(headers: list[str], rows: list[list[str]], *, min_widths: list[int] | None = None) -> str:
    """Render a formatted ASCII table."""
    if not rows:
        return "  (no data)\n"

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    if min_widths:
        for i, mw in enumerate(min_widths):
            if i < len(widths):
                widths[i] = max(widths[i], mw)

    # Build table
    lines: list[str] = []
    sep = "  " + "+".join("-" * (w + 2) for w in widths) + ""

    # Header
    header_line = "  " + "|".join(f" {_bold(h.ljust(w))} " for h, w in zip(headers, widths)) + ""
    lines.append(sep)
    lines.append(header_line)
    lines.append(sep)

    # Rows
    for row in rows:
        cells = []
        for i, (cell, w) in enumerate(zip(row, widths)):
            cells.append(f" {cell.ljust(w)} ")
        lines.append("  " + "|".join(cells) + "")
    lines.append(sep)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Status badges
# ---------------------------------------------------------------------------

def _status_badge(blocked: bool) -> str:
    if blocked:
        return _error("BLOCKED")
    return _success("OK")


def _event_type_badge(event_type: str) -> str:
    badges = {
        "llm_call": _c("LLM", C.BLUE),
        "tool_call": _c("TOOL", C.MAGENTA),
        "policy_violation": _c("VIOLATION", C.RED),
        "escalation": _c("ESCALATION", C.YELLOW),
    }
    return badges.get(event_type, _dim(event_type))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_runs(reader: AuditReader, args: argparse.Namespace) -> None:
    """List all runs with summary stats."""
    runs = reader.get_all_runs()
    if not runs:
        print(_warn("  No runs found in audit log."))
        return

    print()
    print(_header("  AGENT RUNS"))
    print(_dim(f"  {len(runs)} run(s) found | {reader.total_events} total events | ${reader.total_cost:.4f} total cost"))
    print()

    # Sort by first event timestamp
    def _sort_key(r: RunTrace) -> str:
        if r.events:
            return r.events[0].get("timestamp", "")
        return ""

    runs.sort(key=_sort_key, reverse=True)

    headers = ["Run ID", "Events", "Tokens In", "Tokens Out", "Cost", "Violations", "Status"]
    rows = []
    for run in runs[:args.limit]:
        t_in, t_out = run.total_tokens
        blocked_count = sum(1 for e in run.events if e.get("blocked"))
        status = _error("HAS VIOLATIONS") if run.has_violations else _success("CLEAN")

        rows.append([
            run.run_id,
            str(len(run.events)),
            f"{t_in:,}",
            f"{t_out:,}",
            f"${run.total_cost:.4f}",
            str(len(run.violations)),
            status,
        ])

    print(_render_table(headers, rows))


def cmd_replay(reader: AuditReader, args: argparse.Namespace) -> None:
    """Replay a specific run step-by-step."""
    run = reader.get_run(args.run_id)
    if not run.events:
        print(_error(f"  No events found for run '{args.run_id}'"))
        print(_dim("  Use 'agentguard runs' to see available run IDs"))
        return

    print()
    _print_rich_trace(run, show_content=not args.no_content, step_delay=args.delay)


def _print_rich_trace(run: RunTrace, *, show_content: bool = True, step_delay: float = 0) -> None:
    """Rich, colorful step-by-step trace."""
    t_in, t_out = run.total_tokens
    violation_count = len(run.violations)

    # Header
    print(_header("  ╔══════════════════════════════════════════════════════════╗"))
    print(_header("  ║") + _bold("  AGENTGUARD RUN TRACE                                   ") + _header("║"))
    print(_header("  ╠══════════════════════════════════════════════════════════╣"))
    print(_header("  ║") + f"  Run ID:     {_bold(run.run_id):<49}" + _header("║"))
    print(_header("  ║") + f"  Events:     {len(run.events):<49}" + _header("║"))
    print(_header("  ║") + f"  Tokens:     {t_in:,} in / {t_out:,} out{' ' * (49 - len(f'{t_in:,} in / {t_out:,} out'))}" + _header("║"))

    cost_str = f"${run.total_cost:.4f}"
    print(_header("  ║") + f"  Cost:       {cost_str:<49}" + _header("║"))

    if violation_count > 0:
        v_str = _error(f"{violation_count} violation(s)")
        print(_header("  ║") + f"  Violations: {v_str}")
    else:
        print(_header("  ║") + f"  Violations: {_success('None')}")

    print(_header("  ╠══════════════════════════════════════════════════════════╣"))

    for i, event in enumerate(run.events, 1):
        if step_delay > 0 and i > 1:
            time.sleep(step_delay)

        event_type = event.get("event_type", "unknown")
        timestamp = event.get("timestamp", "")
        if isinstance(timestamp, str) and len(timestamp) > 19:
            timestamp = timestamp[:19]

        print(_header("  ║"))

        # -- LLM Call --
        if event_type == "llm_call":
            blocked = event.get("blocked", False)
            model = event.get("model", "unknown")
            tokens_in = event.get("tokens_in", 0)
            tokens_out = event.get("tokens_out", 0)
            cost = event.get("cost_usd", 0)

            badge = _status_badge(blocked)
            print(_header("  ║") + f"  {_bold(f'Step {i}')}  {_event_type_badge(event_type)}  {badge}  {_dim(timestamp)}")
            print(_header("  ║") + f"    Model:  {_c(model, C.CYAN)}")
            print(_header("  ║") + f"    Tokens: {tokens_in:,} in / {tokens_out:,} out  |  Cost: ${cost:.4f}")

            if show_content:
                messages = event.get("messages", [])
                if messages:
                    last = messages[-1]
                    role = last.get("role", "")
                    content = last.get("content", "")
                    if isinstance(content, str) and content:
                        truncated = content[:300] + ("..." if len(content) > 300 else "")
                        print(_header("  ║") + f"    {_dim(role + ':')} {truncated}")

                response = event.get("response_content", "")
                if response:
                    truncated = response[:300] + ("..." if len(response) > 300 else "")
                    print(_header("  ║") + f"    {_dim('assistant:')} {truncated}")

            if blocked:
                reason = event.get("block_reason", "")
                print(_header("  ║") + f"    {_error('BLOCKED: ' + reason)}")

            # PII
            pii_in = event.get("input_pii_found", [])
            pii_out = event.get("output_pii_found", [])
            if pii_in:
                types = [p.get("type", "?") for p in pii_in]
                print(_header("  ║") + f"    {_warn('Input PII: ' + ', '.join(types))}")
            if pii_out:
                types = [p.get("type", "?") for p in pii_out]
                print(_header("  ║") + f"    {_warn('Output PII: ' + ', '.join(types))}")

        # -- Tool Call --
        elif event_type == "tool_call":
            blocked = event.get("blocked", False)
            tool = event.get("tool_name", "unknown")
            duration = event.get("duration_ms", 0)

            badge = _status_badge(blocked)
            print(_header("  ║") + f"  {_bold(f'Step {i}')}  {_event_type_badge(event_type)}  {badge}  {_dim(timestamp)}")
            print(_header("  ║") + f"    Tool:     {_c(tool, C.MAGENTA)}")

            if show_content:
                args_dict = event.get("arguments", {})
                if args_dict:
                    args_str = json.dumps(args_dict, default=str)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(_header("  ║") + f"    Args:     {args_str}")

                result = event.get("result")
                if result is not None:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    print(_header("  ║") + f"    Result:   {result_str}")

            if duration:
                print(_header("  ║") + f"    Duration: {duration:.0f}ms")

            if blocked:
                reason = event.get("block_reason", "")
                print(_header("  ║") + f"    {_error('BLOCKED: ' + reason)}")

        # -- Policy Violation --
        elif event_type == "policy_violation":
            policy = event.get("policy_name", "unknown")
            reason = event.get("violation_reason", "")
            action = event.get("action_taken", "block")

            print(_header("  ║") + f"  {_bold(f'Step {i}')}  {_event_type_badge(event_type)}  {_dim(timestamp)}")
            print(_header("  ║") + f"    Policy: {_error(policy)}")
            print(_header("  ║") + f"    Reason: {reason}")
            print(_header("  ║") + f"    Action: {_warn(action)}")

        # -- Escalation --
        elif event_type == "escalation":
            reason = event.get("reason", "")
            print(_header("  ║") + f"  {_bold(f'Step {i}')}  {_event_type_badge(event_type)}  {_dim(timestamp)}")
            print(_header("  ║") + f"    Reason: {_warn(reason)}")

        else:
            print(_header("  ║") + f"  {_bold(f'Step {i}')}  {_dim(event_type)}  {_dim(timestamp)}")

        # Policy results
        for pr in event.get("policy_results", []):
            if pr.get("action") != "allow":
                print(_header("  ║") + f"    {_dim('Policy')}: {pr.get('policy', '')} -> {_warn(pr.get('action', ''))} ({pr.get('reason', '')})")

    print(_header("  ║"))
    print(_header("  ╚══════════════════════════════════════════════════════════╝"))
    print()


def cmd_violations(reader: AuditReader, args: argparse.Namespace) -> None:
    """Show all policy violations."""
    violations = reader.get_violations()
    if not violations:
        print(_success("\n  No policy violations found."))
        return

    print()
    print(_header(f"  POLICY VIOLATIONS ({len(violations)} total)"))
    print()

    headers = ["#", "Run ID", "Policy", "Reason", "Action", "Timestamp"]
    rows = []
    for i, v in enumerate(violations[:args.limit], 1):
        ts = v.get("timestamp", "")
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]
        rows.append([
            str(i),
            v.get("run_id", "")[:16],
            _error(v.get("policy_name", "")),
            v.get("violation_reason", "")[:60],
            _warn(v.get("action_taken", "")),
            ts,
        ])

    print(_render_table(headers, rows))


def cmd_stats(reader: AuditReader, args: argparse.Namespace) -> None:
    """Show dashboard overview."""
    runs = reader.get_all_runs()
    all_events = reader.get_events(limit=999999)
    violations = reader.get_violations()

    total_tokens_in = sum(e.get("tokens_in", 0) for e in all_events)
    total_tokens_out = sum(e.get("tokens_out", 0) for e in all_events)
    total_cost = reader.total_cost

    # Model breakdown
    model_costs: dict[str, float] = {}
    model_calls: dict[str, int] = {}
    for e in all_events:
        if e.get("event_type") == "llm_call":
            model = e.get("model", "unknown")
            model_costs[model] = model_costs.get(model, 0) + e.get("cost_usd", 0)
            model_calls[model] = model_calls.get(model, 0) + 1

    # Tool breakdown
    tool_calls: dict[str, int] = {}
    tool_blocked: dict[str, int] = {}
    for e in all_events:
        if e.get("event_type") == "tool_call":
            tool = e.get("tool_name", "unknown")
            tool_calls[tool] = tool_calls.get(tool, 0) + 1
            if e.get("blocked"):
                tool_blocked[tool] = tool_blocked.get(tool, 0) + 1

    # Policy breakdown
    policy_counts: dict[str, int] = {}
    for v in violations:
        p = v.get("policy_name", "unknown")
        policy_counts[p] = policy_counts.get(p, 0) + 1

    # Print dashboard
    print()
    print(_header("  ╔══════════════════════════════════════════════════════════╗"))
    print(_header("  ║") + _bold("  AGENTGUARD DASHBOARD                                   ") + _header("║"))
    print(_header("  ╠══════════════════════════════════════════════════════════╣"))
    print()

    # Overview
    print(_bold("  Overview"))
    print(f"  Total Runs:       {len(runs)}")
    print(f"  Total Events:     {reader.total_events}")
    print(f"  Total Violations: {_error(str(len(violations))) if violations else _success('0')}")
    blocked_runs = sum(1 for r in runs if r.has_violations)
    clean_runs = len(runs) - blocked_runs
    print(f"  Clean Runs:       {_success(str(clean_runs))}")
    print(f"  Blocked Runs:     {_error(str(blocked_runs))}")
    print()

    # Cost breakdown
    print(_bold("  Cost Summary"))
    print(f"  Total Cost:       ${total_cost:.4f}")
    print(f"  Tokens In:        {total_tokens_in:,}")
    print(f"  Tokens Out:       {total_tokens_out:,}")
    print()

    if model_costs:
        print(_bold("  Cost by Model"))
        headers = ["Model", "Calls", "Cost"]
        rows = []
        for model in sorted(model_costs, key=model_costs.get, reverse=True):
            rows.append([
                _c(model, C.CYAN),
                str(model_calls.get(model, 0)),
                f"${model_costs[model]:.4f}",
            ])
        print(_render_table(headers, rows))

    if tool_calls:
        print(_bold("  Tool Usage"))
        headers = ["Tool", "Calls", "Blocked"]
        rows = []
        for tool in sorted(tool_calls, key=tool_calls.get, reverse=True):
            blocked = tool_blocked.get(tool, 0)
            rows.append([
                _c(tool, C.MAGENTA),
                str(tool_calls[tool]),
                _error(str(blocked)) if blocked else _success("0"),
            ])
        print(_render_table(headers, rows))

    if policy_counts:
        print(_bold("  Violations by Policy"))
        headers = ["Policy", "Count"]
        rows = []
        for p in sorted(policy_counts, key=policy_counts.get, reverse=True):
            rows.append([_error(p), str(policy_counts[p])])
        print(_render_table(headers, rows))


def cmd_search(reader: AuditReader, args: argparse.Namespace) -> None:
    """Search events by content."""
    query = args.query.lower()
    all_events = reader.get_events(limit=999999)

    matches = []
    for e in all_events:
        text = json.dumps(e, default=str).lower()
        if query in text:
            matches.append(e)

    if not matches:
        print(_warn(f"\n  No events matching '{args.query}'"))
        return

    print()
    print(_header(f"  SEARCH RESULTS: '{args.query}' ({len(matches)} matches)"))
    print()

    headers = ["#", "Run ID", "Type", "Summary", "Timestamp"]
    rows = []
    for i, e in enumerate(matches[:args.limit], 1):
        event_type = e.get("event_type", "unknown")
        ts = e.get("timestamp", "")[:19]

        if event_type == "llm_call":
            summary = f"Model: {e.get('model', '?')}"
        elif event_type == "tool_call":
            summary = f"Tool: {e.get('tool_name', '?')}"
        elif event_type == "policy_violation":
            summary = f"{e.get('policy_name', '?')}: {e.get('violation_reason', '')[:40]}"
        else:
            summary = event_type

        rows.append([
            str(i),
            e.get("run_id", "")[:16],
            _event_type_badge(event_type),
            summary[:50],
            ts,
        ])

    print(_render_table(headers, rows))


def cmd_export(reader: AuditReader, args: argparse.Namespace) -> None:
    """Export events to JSON or CSV."""
    events = reader.get_events(limit=999999)

    if args.run_id:
        events = [e for e in events if e.get("run_id") == args.run_id]

    if not events:
        print(_warn("  No events to export."))
        return

    output_path = args.output

    if args.format == "json":
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, default=str)
            print(_success(f"  Exported {len(events)} events to {output_path}"))
        else:
            print(json.dumps(events, indent=2, default=str))

    elif args.format == "csv":
        if not events:
            return

        # Collect all unique keys
        all_keys: list[str] = []
        seen: set[str] = set()
        for e in events:
            for k in e.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        output = io.StringIO() if not output_path else None
        target = open(output_path, "w", newline="", encoding="utf-8") if output_path else output

        writer = csv.DictWriter(target, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            # Flatten complex fields
            flat = {}
            for k, v in e.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, default=str)
                else:
                    flat[k] = v
            writer.writerow(flat)

        if output_path:
            target.close()
            print(_success(f"  Exported {len(events)} events to {output_path}"))
        else:
            print(output.getvalue())


def cmd_tail(reader: AuditReader, args: argparse.Namespace) -> None:
    """Live tail — watch for new events."""
    print(_header("\n  LIVE TAIL — watching for new events (Ctrl+C to stop)"))
    print(_dim(f"  File: {reader._path}"))
    print()

    last_count = reader.total_events

    try:
        while True:
            reader.reload()
            current_count = reader.total_events

            if current_count > last_count:
                new_events = reader.get_events(limit=999999)[last_count:]
                for e in new_events:
                    event_type = e.get("event_type", "unknown")
                    ts = e.get("timestamp", "")[:19]
                    run_id = e.get("run_id", "")[:12]
                    blocked = e.get("blocked", False)

                    badge = _event_type_badge(event_type)
                    status = _error(" BLOCKED") if blocked else ""

                    if event_type == "llm_call":
                        detail = f"model={e.get('model', '?')} tokens={e.get('tokens_in', 0)}+{e.get('tokens_out', 0)}"
                    elif event_type == "tool_call":
                        detail = f"tool={e.get('tool_name', '?')}"
                    elif event_type == "policy_violation":
                        detail = f"policy={e.get('policy_name', '?')} reason={e.get('violation_reason', '')[:50]}"
                    else:
                        detail = ""

                    print(f"  {_dim(ts)}  {badge}  {_dim(run_id)}  {detail}{status}")

                last_count = current_count

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(_dim("\n  Stopped."))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentguard",
        description="AgentGuard CLI — audit reader and forensics tool",
    )
    parser.add_argument(
        "-f", "--file",
        default="agentguard_audit.jsonl",
        help="Path to the audit log file (default: agentguard_audit.jsonl)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- runs --
    p_runs = subparsers.add_parser("runs", help="List all agent runs")
    p_runs.add_argument("-n", "--limit", type=int, default=50, help="Max runs to show")

    # -- replay --
    p_replay = subparsers.add_parser("replay", help="Replay a specific run step-by-step")
    p_replay.add_argument("run_id", help="Run ID to replay")
    p_replay.add_argument("--no-content", action="store_true", help="Hide message content")
    p_replay.add_argument("--delay", type=float, default=0, help="Delay between steps (seconds)")

    # -- violations --
    p_violations = subparsers.add_parser("violations", help="Show all policy violations")
    p_violations.add_argument("-n", "--limit", type=int, default=100, help="Max violations")

    # -- stats --
    subparsers.add_parser("stats", help="Show dashboard overview")

    # -- search --
    p_search = subparsers.add_parser("search", help="Search events by content")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=50, help="Max results")

    # -- export --
    p_export = subparsers.add_parser("export", help="Export events to JSON or CSV")
    p_export.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    p_export.add_argument("--output", "-o", help="Output file path (default: stdout)")
    p_export.add_argument("--run-id", help="Filter to specific run")

    # -- tail --
    p_tail = subparsers.add_parser("tail", help="Live tail — watch for new events")
    p_tail.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    # Enable UTF-8 on Windows (only when running as actual CLI, not in pytest)
    if sys.platform == "win32" and not hasattr(sys, "_called_from_test"):
        try:
            import io as _io
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    reader = AuditReader(args.file)

    commands = {
        "runs": cmd_runs,
        "replay": cmd_replay,
        "violations": cmd_violations,
        "stats": cmd_stats,
        "search": cmd_search,
        "export": cmd_export,
        "tail": cmd_tail,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(reader, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
