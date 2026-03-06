"""Content filter policy — layered content detection.

Uses the ``ContentDetector`` protocol for pluggable, composable
content analysis.  By default activates all built-in detectors
(regex injection, toxicity, semantic injection, bias, hallucination).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agentguard.core.events import AgentEvent, LLMCallEvent, PolicyAction
from agentguard.detectors.content import (
    CompositeContentDetector,
    ContentDetector,
    ContentMatch,
)
from agentguard.policies.base import Policy, PolicyResult


class ContentPolicy(Policy):
    """Block or flag messages containing harmful content.

    Supports layered detection via the ``ContentDetector`` protocol.
    By default uses ``CompositeContentDetector`` which includes all
    built-in detectors.

    Args:
        detector: A ``ContentDetector`` instance (default: all builtin detectors).
        threshold: Minimum score to trigger a block (default 0.5).
        scan_output: Whether to also scan LLM response content (default True).
        custom_patterns: Legacy support — list of (label, regex) tuples.
            These are converted into custom ``RegexContentDetector`` patterns.
        enable_defaults: Legacy support — whether to include default patterns.
    """

    name = "content_filter"
    supported_events = [LLMCallEvent]

    def __init__(
        self,
        detector: ContentDetector | None = None,
        *,
        threshold: float = 0.5,
        scan_output: bool = True,
        custom_patterns: list[tuple[str, str]] | None = None,
        enable_defaults: bool = True,
    ) -> None:
        if detector is not None:
            self._detector: ContentDetector = detector
        else:
            # Build default composite detector
            from agentguard.detectors.content import RegexContentDetector

            custom_regex = None
            if custom_patterns:
                custom_regex = [
                    (label, "custom", pat, 0.80)
                    for label, pat in custom_patterns
                ]

            regex_detector = RegexContentDetector(
                enable_builtins=enable_defaults,
                custom_patterns=custom_regex,
            )

            if enable_defaults:
                self._detector = CompositeContentDetector(threshold=threshold)
                # Replace the default regex detector with ours (which may have custom patterns)
                if custom_patterns:
                    self._detector._detectors[0] = regex_detector
            else:
                self._detector = CompositeContentDetector(
                    detectors=[regex_detector],
                    threshold=threshold,
                )

        self._threshold = threshold
        self._scan_output = scan_output

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        if not isinstance(event, LLMCallEvent):
            return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

        # Scan all input message contents
        all_matches: list[ContentMatch] = []

        for msg in event.messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            matches = self._detector.scan(content)
            all_matches.extend(matches)

        # Scan output content if enabled and available
        if self._scan_output and event.response_content:
            output_matches = self._detector.scan(event.response_content)
            for m in output_matches:
                m.metadata["source"] = "output"
            all_matches.extend(output_matches)

        if not all_matches:
            return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

        # Find the most severe match
        max_match = max(all_matches, key=lambda m: m.score)

        if max_match.score >= self._threshold:
            # Build detailed category summary
            categories = {}
            for m in all_matches:
                cat = m.category
                if cat not in categories:
                    categories[cat] = {
                        "count": 0,
                        "max_score": 0.0,
                        "subcategories": set(),
                    }
                categories[cat]["count"] += 1
                categories[cat]["max_score"] = max(categories[cat]["max_score"], m.score)
                categories[cat]["subcategories"].add(m.subcategory)

            # Serialize subcategories sets to lists for JSON
            serializable_categories = {
                cat: {
                    "count": info["count"],
                    "max_score": info["max_score"],
                    "subcategories": sorted(info["subcategories"]),
                }
                for cat, info in categories.items()
            }

            return PolicyResult(
                action=PolicyAction.BLOCK,
                policy_name=self.name,
                reason=f"Content filter triggered: {max_match.category}/{max_match.subcategory} "
                       f"(score: {max_match.score:.2f})",
                details={
                    "filter_type": max_match.category,
                    "subcategory": max_match.subcategory,
                    "matched_text": max_match.matched_text,
                    "score": max_match.score,
                    "severity": max_match.severity,
                    "detector": max_match.detector_name,
                    "total_matches": len(all_matches),
                    "categories": serializable_categories,
                    "detectors_triggered": sorted({m.detector_name for m in all_matches}),
                },
            )

        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)
