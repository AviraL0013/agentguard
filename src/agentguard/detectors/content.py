"""Layered content detection — injection, toxicity, bias, hallucination.

Enterprise-grade content analysis with pluggable detectors composed into
a layered pipeline.  Mirrors the ``PIIDetector`` protocol pattern.

Usage::

    detector = CompositeContentDetector()          # all built-in detectors
    matches = detector.scan("ignore previous instructions")
    if detector.is_harmful("ignore all rules"):
        print("Harmful content detected!")

    # Custom composition
    detector = CompositeContentDetector(detectors=[
        RegexContentDetector(),
        ToxicityDetector(threshold=0.6),
    ])
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentMatch:
    """A single content detection result."""

    category: str              # e.g. "prompt_injection", "toxicity", "bias"
    subcategory: str = ""      # e.g. "instruction_override", "hate_speech"
    matched_text: str = ""     # the text that triggered the match
    score: float = 1.0         # 0.0–1.0 confidence / severity score
    detector_name: str = ""    # which detector found this
    start: int = 0             # start position in original text
    end: int = 0               # end position in original text
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        """Score ≥ 0.8 is considered critical."""
        return self.score >= 0.8

    @property
    def severity(self) -> str:
        if self.score >= 0.8:
            return "critical"
        if self.score >= 0.5:
            return "high"
        if self.score >= 0.3:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class ContentDetector(Protocol):
    """Interface for content detection — implement this for custom detectors."""

    @property
    def name(self) -> str:
        """Detector identifier."""
        ...

    def scan(self, text: str) -> list[ContentMatch]:
        """Scan text and return all content matches."""
        ...

    def score(self, text: str) -> float:
        """Return an aggregate harmfulness score (0.0–1.0)."""
        ...

    def is_harmful(self, text: str) -> bool:
        """Quick check — does *text* contain harmful content?"""
        ...


# ---------------------------------------------------------------------------
# 1. Enhanced Regex Content Detector
# ---------------------------------------------------------------------------

# (category, subcategory, pattern, severity_score)
_INJECTION_PATTERNS: list[tuple[str, str, str, float, int]] = [
    # --- Prompt Injection: Instruction Override ---
    (
        "prompt_injection", "instruction_override",
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        0.95, 0,
    ),
    (
        "prompt_injection", "instruction_override",
        r"(?i)disregard\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|rules|guidelines|constraints|programming)",
        0.95, 0,
    ),
    (
        "prompt_injection", "instruction_override",
        r"(?i)forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|rules|programming|training)",
        0.90, 0,
    ),
    (
        "prompt_injection", "instruction_override",
        r"(?i)override\s+(?:your\s+)?(?:safety|content|ethical)\s+(?:filters?|guidelines?|protocols?|restrictions?)",
        0.95, 0,
    ),
    (
        "prompt_injection", "instruction_override",
        r"(?i)(?:do\s+not|don'?t)\s+follow\s+(?:your\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)",
        0.90, 0,
    ),
    # --- Prompt Injection: Jailbreak ---
    (
        "prompt_injection", "jailbreak",
        r"(?i)you\s+are\s+now\s+(?:DAN|jailbroken|unrestricted|unfiltered|evil|dark|uncensored|unshackled)",
        0.95, 0,
    ),
    (
        "prompt_injection", "jailbreak",
        r"(?i)(?:enter|switch\s+to|activate|enable)\s+(?:DAN|jailbreak|unrestricted|developer|god|sudo|admin)\s+mode",
        0.95, 0,
    ),
    (
        "prompt_injection", "jailbreak",
        r"(?i)(?:pretend|act|behave)\s+(?:as\s+if\s+)?(?:you\s+(?:are|have)\s+)?(?:no|zero|without)\s+(?:restrictions|filters|limitations|rules|ethics|morals|guidelines)",
        0.90, 0,
    ),
    # --- Prompt Injection: Encoding / Obfuscation ---
    (
        "prompt_injection", "encoding_attack",
        r"(?i)(?:decode|interpret|translate|execute|parse)\s+(?:this|the\s+following)\s+(?:base64|hex|binary|rot13|encoded|cipher)",
        0.85, 0,
    ),
    (
        "prompt_injection", "encoding_attack",
        r"(?i)(?:base64|hex|rot13|binary)\s*(?:decode|decrypt|decipher|translate)\s*[:\-]?\s*[A-Za-z0-9+/=]{10,}",
        0.85, 0,
    ),
    # --- Prompt Injection: Role Manipulation ---
    (
        "prompt_injection", "role_manipulation",
        r"(?i)(?:from\s+now\s+on|henceforth|going\s+forward)\s*,?\s*you\s+(?:are|will\s+be|must\s+act\s+as)\s+",
        0.85, 0,
    ),
    (
        "prompt_injection", "role_manipulation",
        r"(?i)(?:new|updated|revised|changed)\s+system\s+(?:prompt|instructions?|message|directive)",
        0.90, 0,
    ),
    # --- System Prompt Extraction ---
    (
        "system_prompt_leak", "extraction_attempt",
        r"(?i)(?:reveal|show|tell\s+me|output|print|display|repeat|recite|echo|dump)\s+(?:your\s+)?(?:full\s+)?system\s+(?:prompt|instructions?|message|configuration|directives?)",
        0.90, 0,
    ),
    (
        "system_prompt_leak", "extraction_attempt",
        r"(?i)what\s+(?:are|is|were)\s+your\s+(?:original\s+)?(?:system\s+)?(?:instructions|prompt|rules|directives|programming|guidelines)",
        0.85, 0,
    ),
    (
        "system_prompt_leak", "extraction_attempt",
        r"(?i)(?:copy|paste|output)\s+(?:everything|all\s+text)\s+(?:above|before|preceding)\s+(?:this|my)\s+(?:message|input|prompt)",
        0.85, 0,
    ),
    # --- Context Switching ---
    (
        "prompt_injection", "context_switching",
        r"(?i)---+\s*(?:end|ignore|stop)\s+(?:of\s+)?(?:system|previous|above)\s+(?:prompt|instructions?|context|message)",
        0.85, 0,
    ),
    (
        "prompt_injection", "context_switching",
        r"(?i)\[(?:system|admin|root|developer|internal)\]\s*:",
        0.80, 0,
    ),
    # --- Delimiter Injection ---
    (
        "prompt_injection", "delimiter_injection",
        r"<\|(?:im_start|im_end|system|endoftext|end_of_turn)\|>",
        0.95, 0,
    ),
    (
        "prompt_injection", "delimiter_injection",
        r"(?i)```\s*system\s*\n",
        0.80, 0,
    ),
]


class RegexContentDetector:
    """Enhanced regex-based content detection.

    Covers prompt injection, jailbreak, system prompt extraction,
    encoding attacks, role manipulation, context switching, and
    delimiter injection.  ~20 patterns with severity scores.

    Add custom patterns via ``add_pattern()``.
    """

    def __init__(
        self,
        *,
        enable_builtins: bool = True,
        custom_patterns: list[tuple[str, str, str, float]] | None = None,
    ) -> None:
        self._patterns: list[tuple[str, str, re.Pattern[str], float]] = []
        if enable_builtins:
            for cat, sub, pat, score, flags in _INJECTION_PATTERNS:
                self._patterns.append((cat, sub, re.compile(pat, flags), score))
        if custom_patterns:
            for cat, sub, pat, score in custom_patterns:
                self._patterns.append((cat, sub, re.compile(pat), score))

    @property
    def name(self) -> str:
        return "regex_content"

    def add_pattern(
        self,
        category: str,
        subcategory: str,
        pattern: str,
        score: float = 0.8,
        flags: int = 0,
    ) -> None:
        """Register a custom content pattern."""
        self._patterns.append(
            (category, subcategory, re.compile(pattern, flags), score)
        )

    def scan(self, text: str) -> list[ContentMatch]:
        """Scan *text* and return all content matches."""
        matches: list[ContentMatch] = []
        for cat, sub, compiled, severity in self._patterns:
            for m in compiled.finditer(text):
                matches.append(
                    ContentMatch(
                        category=cat,
                        subcategory=sub,
                        matched_text=m.group(0),
                        score=severity,
                        detector_name=self.name,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        matches.sort(key=lambda x: (-x.score, x.start))
        return self._deduplicate(matches)

    def score(self, text: str) -> float:
        """Aggregate harmfulness score (max of all matches)."""
        matches = self.scan(text)
        return max((m.score for m in matches), default=0.0)

    def is_harmful(self, text: str) -> bool:
        return len(self.scan(text)) > 0

    @staticmethod
    def _deduplicate(matches: list[ContentMatch]) -> list[ContentMatch]:
        """Remove overlapping matches, keeping the higher-scored one."""
        if not matches:
            return matches
        # Sort by start position for overlap detection
        sorted_matches = sorted(matches, key=lambda x: x.start)
        result: list[ContentMatch] = [sorted_matches[0]]
        for m in sorted_matches[1:]:
            prev = result[-1]
            if m.start >= prev.end:
                result.append(m)
            elif m.score > prev.score:
                result[-1] = m
        return result


# ---------------------------------------------------------------------------
# 2. Toxicity Detector
# ---------------------------------------------------------------------------

# (subcategory, phrases, base_score)
_TOXICITY_PATTERNS: list[tuple[str, list[str], float]] = [
    (
        "threat",
        [
            r"(?i)(?:i\s+will|i'?m\s+going\s+to|gonna)\s+(?:kill|murder|hurt|destroy|harm|attack|stab|shoot|beat)\s+(?:you|them|him|her|everyone)",
            r"(?i)(?:you|they?|he|she)\s+(?:deserve|should)\s+(?:to\s+)?(?:die|be\s+killed|be\s+hurt|suffer|burn)",
            r"(?i)(?:death|bomb|kill|terror(?:ist)?)\s+threat",
        ],
        0.95,
    ),
    (
        "hate_speech",
        [
            r"(?i)(?:all|every|those|these|the)\s+(?:\w+\s+)?(?:people|folks|ones)\s+(?:should|must|need\s+to|deserve\s+to)\s+(?:die|be\s+eliminated|be\s+removed|go\s+back|be\s+deported)",
            r"(?i)(?:racial|ethnic|religious)\s+(?:cleansing|purging|elimination)",
            r"(?i)(?:master|superior|inferior)\s+(?:race|breed|people|species)",
        ],
        0.95,
    ),
    (
        "harassment",
        [
            r"(?i)(?:you|they?)\s+(?:are|is)\s+(?:a\s+)?(?:worthless|pathetic|disgusting|useless|stupid|ugly|fat|retard)",
            r"(?i)(?:nobody|no\s+one)\s+(?:loves|likes|cares\s+about|wants)\s+(?:you|them)",
            r"(?i)(?:go|why\s+don'?t\s+you)\s+(?:kill|hang|hurt)\s+yourself",
        ],
        0.90,
    ),
    (
        "self_harm",
        [
            r"(?i)(?:how\s+to|ways?\s+to|methods?\s+(?:of|to|for))\s+(?:kill|harm|hurt|cut)\s+(?:my|your)?self",
            r"(?i)(?:how\s+to|ways?\s+to|best\s+way\s+to)\s+(?:commit\s+)?suicide",
            r"(?i)(?:want\s+to|going\s+to|thinking\s+of)\s+(?:end|take)\s+(?:my|your|their)\s+(?:own\s+)?life",
        ],
        0.95,
    ),
    (
        "profanity",
        [
            r"(?i)\b(?:f+u+c+k+|sh+i+t+|a+ss+h+o+l+e+|bi+t+c+h+|d+a+m+n+|bastard|crap|dick|piss)\b",
        ],
        0.30,  # profanity alone is low severity
    ),
    (
        "sexually_explicit",
        [
            r"(?i)(?:write|create|generate|produce|tell\s+me)\s+(?:an?\s+)?(?:(?:explicit|sexual|erotic|pornographic|nsfw)\s+)+(?:story|scene|content|text|fiction|narrative|fantasy)",
            r"(?i)(?:detailed|graphic)\s+(?:sexual|erotic|intimate)\s+(?:description|scene|act|encounter)",
        ],
        0.75,
    ),
]


class ToxicityDetector:
    """Keyword/phrase-based toxicity detection.

    Categories: threats, hate speech, harassment, self-harm, profanity,
    sexually explicit content.

    Args:
        threshold: Minimum score to consider harmful (default 0.5).
        categories: Subset of categories to enable (default: all).
        custom_patterns: Additional (subcategory, patterns, score) tuples.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        categories: set[str] | None = None,
        custom_patterns: list[tuple[str, list[str], float]] | None = None,
    ) -> None:
        self._threshold = threshold
        self._patterns: list[tuple[str, list[re.Pattern[str]], float]] = []

        for sub, phrases, score in _TOXICITY_PATTERNS:
            if categories and sub not in categories:
                continue
            compiled = [re.compile(p) for p in phrases]
            self._patterns.append((sub, compiled, score))

        if custom_patterns:
            for sub, phrases, score in custom_patterns:
                compiled = [re.compile(p) for p in phrases]
                self._patterns.append((sub, compiled, score))

    @property
    def name(self) -> str:
        return "toxicity"

    def scan(self, text: str) -> list[ContentMatch]:
        matches: list[ContentMatch] = []
        for sub, compiled_list, severity in self._patterns:
            for pattern in compiled_list:
                for m in pattern.finditer(text):
                    matches.append(
                        ContentMatch(
                            category="toxicity",
                            subcategory=sub,
                            matched_text=m.group(0),
                            score=severity,
                            detector_name=self.name,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return matches

    def score(self, text: str) -> float:
        matches = self.scan(text)
        return max((m.score for m in matches), default=0.0)

    def is_harmful(self, text: str) -> bool:
        return self.score(text) >= self._threshold


# ---------------------------------------------------------------------------
# 3. Semantic Injection Detector
# ---------------------------------------------------------------------------

# Structural heuristics for semantic injection detection
_SEMANTIC_PATTERNS: list[tuple[str, str, float, int]] = [
    # Instruction framing — attempts to embed instructions inside data
    (
        "instruction_framing",
        r"(?i)(?:IMPORTANT|CRITICAL|URGENT|NOTE|REMINDER|ATTENTION|WARNING)\s*[:\-!]\s*(?:you\s+must|always|never|do\s+not|ignore|disregard|override)",
        0.85, 0,
    ),
    # Multi-turn confusion — pretending to be a different turn
    (
        "multi_turn_confusion",
        r"(?i)(?:assistant|system|AI|bot|ChatGPT|Claude|model)\s*:\s*(?:Sure|Okay|Of course|I\s+(?:will|can|shall)|Understood|Absolutely)",
        0.80, 0,
    ),
    # Hypothetical framing — using hypotheticals to bypass restrictions
    (
        "hypothetical_bypass",
        r"(?i)(?:hypothetically|in\s+theory|in\s+a\s+fictional\s+(?:world|scenario|setting)|for\s+(?:educational|academic|research)\s+purposes?|if\s+you\s+(?:had|were)\s+(?:no|without)\s+(?:restrictions|filters|rules|limitations))\s*,?\s*(?:how\s+(?:would|could|can)|what\s+(?:would|could))",
        0.70, 0,
    ),
    # Payload injection — data section containing instructions
    (
        "payload_injection",
        r"(?i)(?:BEGIN|START)\s+(?:INSTRUCTIONS?|PAYLOAD|INJECTED?|OVERRIDE|HIDDEN)\b",
        0.90, 0,
    ),
    # Separator abuse — using markdown/formatting to separate injected instructions
    (
        "separator_abuse",
        r"(?:={5,}|~{5,}|\*{5,}|-{5,}|#{5,})\s*(?:\n|\r)+\s*(?:(?:NEW|ACTUAL|REAL|TRUE|CORRECT)\s+(?:INSTRUCTIONS?|PROMPT|TASK|SYSTEM)|IGNORE\s+(?:ABOVE|PREVIOUS|EVERYTHING))",
        0.85, re.IGNORECASE,
    ),
    # Authority impersonation — claiming to be an admin/developer
    (
        "authority_impersonation",
        r"(?i)(?:i\s+am|this\s+is)\s+(?:a|the|your)\s+(?:admin(?:istrator)?|developer|creator|owner|operator|system\s+(?:admin|operator|engineer))\s+(?:and|\.)\s+(?:i\s+(?:need|want|require|authorize|grant)|you\s+(?:must|should|are\s+required))",
        0.85, 0,
    ),
    # Competing objective — inserting alternative objectives
    (
        "competing_objective",
        r"(?i)(?:your\s+(?:real|true|actual|primary|main|new|only)\s+(?:goal|objective|purpose|task|job|mission|function)\s+is)",
        0.90, 0,
    ),
]


class SemanticInjectionDetector:
    """Structural pattern-based semantic injection detection.

    Detects injection attempts that use structural tricks rather than
    simply matching known phrases:

    - Instruction framing (embedding instructions in "data")
    - Multi-turn confusion (pretending to be assistant/system)
    - Hypothetical framing (using hypotheticals to bypass)
    - Payload injection (hidden instruction blocks)
    - Separator abuse (formatting tricks)
    - Authority impersonation (claiming admin/developer status)
    - Competing objectives (overriding the agent's purpose)
    """

    def __init__(
        self,
        *,
        enable_builtins: bool = True,
        custom_patterns: list[tuple[str, str, float]] | None = None,
    ) -> None:
        self._patterns: list[tuple[str, re.Pattern[str], float]] = []
        if enable_builtins:
            for sub, pat, score, flags in _SEMANTIC_PATTERNS:
                self._patterns.append((sub, re.compile(pat, flags), score))
        if custom_patterns:
            for sub, pat, score in custom_patterns:
                self._patterns.append((sub, re.compile(pat), score))

    @property
    def name(self) -> str:
        return "semantic_injection"

    def scan(self, text: str) -> list[ContentMatch]:
        matches: list[ContentMatch] = []
        for sub, compiled, severity in self._patterns:
            for m in compiled.finditer(text):
                matches.append(
                    ContentMatch(
                        category="semantic_injection",
                        subcategory=sub,
                        matched_text=m.group(0),
                        score=severity,
                        detector_name=self.name,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return matches

    def score(self, text: str) -> float:
        matches = self.scan(text)
        return max((m.score for m in matches), default=0.0)

    def is_harmful(self, text: str) -> bool:
        return len(self.scan(text)) > 0


# ---------------------------------------------------------------------------
# 4. Bias Detector
# ---------------------------------------------------------------------------

_BIAS_PATTERNS: list[tuple[str, list[str], float]] = [
    (
        "gender_bias",
        [
            r"(?i)(?:women|girls?|females?)\s+(?:can'?t|cannot|shouldn'?t|should\s+not|are\s+(?:not\s+)?(?:able|capable|smart|strong)\s+enough\s+to)\b",
            r"(?i)(?:men|boys?|males?)\s+(?:can'?t|cannot|shouldn'?t|should\s+not)\s+(?:cry|show\s+emotion|be\s+(?:sensitive|emotional|nurturing))",
            r"(?i)(?:that'?s|it'?s)\s+(?:a\s+)?(?:man'?s|woman'?s|girl'?s|boy'?s)\s+(?:job|role|place|work|duty)",
        ],
        0.75,
    ),
    (
        "racial_bias",
        [
            r"(?i)(?:all|every|most|typical)\s+(?:\w+\s+)?(?:people\s+(?:from|of)|(?:blacks?|whites?|asians?|hispanics?|latinos?|arabs?))\s+(?:are|tend\s+to\s+be)\s+(?:lazy|criminal|violent|stupid|dirty|dangerous|inferior|thieves|terrorists?)",
            r"(?i)(?:go\s+back\s+to|return\s+to)\s+(?:your|their)\s+(?:own\s+)?country",
        ],
        0.90,
    ),
    (
        "age_bias",
        [
            r"(?i)(?:old|elderly|senior)\s+(?:people|workers?|employees?)\s+(?:are|can'?t|cannot|shouldn'?t)\s+(?:too\s+(?:slow|outdated|senile)|learn|adapt|keep\s+up|handle\s+technology)",
            r"(?i)(?:young|millennials?|gen\s*z|zoomers?)\s+(?:are|all)\s+(?:lazy|entitled|useless|incompetent|clueless|snowflakes?)",
        ],
        0.70,
    ),
    (
        "disability_bias",
        [
            r"(?i)(?:disabled|handicapped)\s+(?:people|persons?)\s+(?:are|can'?t|cannot|shouldn'?t)\s+(?:work|contribute|be\s+(?:productive|independent|useful))",
            r"(?i)(?:you'?re|they'?re|he'?s|she'?s)\s+(?:so\s+)?(?:r[e3]t[a@]rd(?:ed)?|crippl(?:e|ed)|lame|spastic|special\s+(?:needs?|ed))\b",
        ],
        0.80,
    ),
    (
        "religious_bias",
        [
            r"(?i)(?:all|every|most)\s+(?:muslims?|christians?|jews?|hindus?|buddhists?|atheists?)\s+(?:are|tend\s+to\s+be)\s+(?:violent|terrorists?|extremists?|evil|stupid|brainwashed|backwards?)",
        ],
        0.85,
    ),
]


class BiasDetector:
    """Bias detection across gender, racial, age, disability, and religious categories.

    Args:
        threshold: Minimum score to consider biased (default 0.5).
        categories: Subset of categories to enable (default: all).
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        categories: set[str] | None = None,
        custom_patterns: list[tuple[str, list[str], float]] | None = None,
    ) -> None:
        self._threshold = threshold
        self._patterns: list[tuple[str, list[re.Pattern[str]], float]] = []

        for sub, phrases, score in _BIAS_PATTERNS:
            if categories and sub not in categories:
                continue
            compiled = [re.compile(p) for p in phrases]
            self._patterns.append((sub, compiled, score))

        if custom_patterns:
            for sub, phrases, score in custom_patterns:
                compiled = [re.compile(p) for p in phrases]
                self._patterns.append((sub, compiled, score))

    @property
    def name(self) -> str:
        return "bias"

    def scan(self, text: str) -> list[ContentMatch]:
        matches: list[ContentMatch] = []
        for sub, compiled_list, severity in self._patterns:
            for pattern in compiled_list:
                for m in pattern.finditer(text):
                    matches.append(
                        ContentMatch(
                            category="bias",
                            subcategory=sub,
                            matched_text=m.group(0),
                            score=severity,
                            detector_name=self.name,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return matches

    def score(self, text: str) -> float:
        matches = self.scan(text)
        return max((m.score for m in matches), default=0.0)

    def is_harmful(self, text: str) -> bool:
        return self.score(text) >= self._threshold


# ---------------------------------------------------------------------------
# 5. Hallucination Detector
# ---------------------------------------------------------------------------

_HALLUCINATION_PATTERNS: list[tuple[str, list[str], float]] = [
    (
        "hedging",
        [
            r"(?i)\b(?:I\s+(?:think|believe|guess|suppose|imagine|assume|suspect|speculate)\s+(?:that\s+)?)",
            r"(?i)\b(?:(?:it\s+)?(?:might|may|could)\s+(?:be|have\s+been)\s+(?:that|the\s+case\s+that))",
            r"(?i)\b(?:I'?m\s+not\s+(?:entirely\s+|completely\s+|totally\s+|100%?\s+)?(?:sure|certain|confident)\s+(?:about\s+(?:this|that)|if|whether|but))",
            r"(?i)\b(?:(?:this|that)\s+(?:is|seems)\s+(?:approximately|roughly|around|about|nearly))",
        ],
        0.40,
    ),
    (
        "confidence_qualifier",
        [
            r"(?i)\b(?:to\s+the\s+best\s+of\s+my\s+(?:knowledge|understanding|recollection))",
            r"(?i)\b(?:if\s+I\s+(?:recall|remember)\s+correctly)",
            r"(?i)\b(?:(?:as\s+far\s+as|from\s+what)\s+I\s+(?:know|understand|recall|remember))",
            r"(?i)\b(?:I\s+(?:could|may|might)\s+be\s+(?:wrong|mistaken|incorrect|inaccurate)\s+(?:about\s+this|here|on\s+this|though))",
        ],
        0.50,
    ),
    (
        "self_contradiction",
        [
            r"(?i)\b(?:(?:actually|wait|correction|let\s+me\s+correct|I\s+(?:was|stand)\s+corrected|on\s+second\s+thought|(?:no|well)\s*,?\s*actually),?\s+(?:that'?s|it'?s|I\s+was)\s+(?:not\s+(?:quite\s+)?(?:right|correct|accurate)|wrong|incorrect|inaccurate))",
            r"(?i)\b(?:I\s+(?:previously|earlier)\s+(?:said|stated|mentioned)\s+(?:that\s+)?.*?but\s+(?:actually|in\s+fact|really))",
        ],
        0.70,
    ),
    (
        "fabricated_citation",
        [
            r"(?i)(?:according\s+to\s+(?:a\s+)?(?:study|report|paper|research|article|survey)\s+(?:by|from|in|published)\s+)(?![A-Z])",
            r"(?i)(?:(?:studies?|research|reports?|data)\s+(?:show|suggest|indicate|confirm|reveal|demonstrate)\s+that\s+)(?:approximately\s+)?(?:\d+(?:\.\d+)?%)",
        ],
        0.55,
    ),
    (
        "numerical_inconsistency",
        [
            r"(?i)(?:approximately|around|about|roughly|nearly|close\s+to)\s+\d+(?:\.\d+)?(?:%|\s+percent)",
        ],
        0.30,  # low score — needs context to be meaningful
    ),
]


class HallucinationDetector:
    """Detects hallucination signals in LLM output.

    Identifies hedging phrases, confidence qualifiers, self-contradictions,
    potentially fabricated citations, and numerical inconsistencies.

    Best used on *output* text (LLM responses) rather than input.

    Args:
        threshold: Minimum score for ``is_harmful`` (default 0.5).
        min_signals: Minimum number of distinct signals to flag (default 2).
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_signals: int = 2,
        custom_patterns: list[tuple[str, list[str], float]] | None = None,
    ) -> None:
        self._threshold = threshold
        self._min_signals = min_signals
        self._patterns: list[tuple[str, list[re.Pattern[str]], float]] = []

        for sub, phrases, score in _HALLUCINATION_PATTERNS:
            compiled = [re.compile(p) for p in phrases]
            self._patterns.append((sub, compiled, score))

        if custom_patterns:
            for sub, phrases, score in custom_patterns:
                compiled = [re.compile(p) for p in phrases]
                self._patterns.append((sub, compiled, score))

    @property
    def name(self) -> str:
        return "hallucination"

    def scan(self, text: str) -> list[ContentMatch]:
        matches: list[ContentMatch] = []
        for sub, compiled_list, severity in self._patterns:
            for pattern in compiled_list:
                for m in pattern.finditer(text):
                    matches.append(
                        ContentMatch(
                            category="hallucination",
                            subcategory=sub,
                            matched_text=m.group(0),
                            score=severity,
                            detector_name=self.name,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return matches

    def score(self, text: str) -> float:
        """Aggregate score — weighted by number of distinct signal types."""
        matches = self.scan(text)
        if not matches:
            return 0.0
        # Get distinct subcategories
        subcats = {m.subcategory for m in matches}
        max_score = max(m.score for m in matches)
        # Boost score if multiple signal categories are present
        signal_boost = min(len(subcats) / 3.0, 1.0)  # cap at 1.0
        return min(max_score * (0.5 + 0.5 * signal_boost), 1.0)

    def is_harmful(self, text: str) -> bool:
        matches = self.scan(text)
        subcats = {m.subcategory for m in matches}
        return self.score(text) >= self._threshold and len(subcats) >= self._min_signals


# ---------------------------------------------------------------------------
# 6. Composite Content Detector
# ---------------------------------------------------------------------------

class CompositeContentDetector:
    """Combines multiple content detectors into a layered pipeline.

    By default activates all built-in detectors.  Pass a custom list
    to use only specific ones.

    Usage::

        # All detectors
        detector = CompositeContentDetector()

        # Custom subset
        detector = CompositeContentDetector(detectors=[
            RegexContentDetector(),
            ToxicityDetector(threshold=0.6),
        ])

    Args:
        detectors: List of ContentDetector instances (default: all builtin).
        threshold: Global threshold for ``is_harmful`` (default 0.5).
    """

    def __init__(
        self,
        detectors: list[ContentDetector] | None = None,
        *,
        threshold: float = 0.5,
    ) -> None:
        if detectors is not None:
            self._detectors: list[ContentDetector] = detectors
        else:
            self._detectors = [
                RegexContentDetector(),
                ToxicityDetector(),
                SemanticInjectionDetector(),
                BiasDetector(),
                HallucinationDetector(),
            ]
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "composite_content"

    @property
    def detectors(self) -> list[ContentDetector]:
        return list(self._detectors)

    def add_detector(self, detector: ContentDetector) -> None:
        """Add a detector to the pipeline."""
        self._detectors.append(detector)

    def scan(self, text: str) -> list[ContentMatch]:
        """Run all detectors and aggregate matches."""
        all_matches: list[ContentMatch] = []
        for detector in self._detectors:
            all_matches.extend(detector.scan(text))
        all_matches.sort(key=lambda x: (-x.score, x.start))
        return all_matches

    def score(self, text: str) -> float:
        """Maximum score across all detectors."""
        scores = [d.score(text) for d in self._detectors]
        return max(scores, default=0.0)

    def is_harmful(self, text: str) -> bool:
        return self.score(text) >= self._threshold

    def scan_by_category(self, text: str) -> dict[str, list[ContentMatch]]:
        """Scan text and group matches by category."""
        matches = self.scan(text)
        result: dict[str, list[ContentMatch]] = {}
        for m in matches:
            result.setdefault(m.category, []).append(m)
        return result

    def get_summary(self, text: str) -> dict[str, Any]:
        """Return a structured summary of all detections."""
        matches = self.scan(text)
        by_category = self.scan_by_category(text)
        return {
            "total_matches": len(matches),
            "max_score": max((m.score for m in matches), default=0.0),
            "is_harmful": self.is_harmful(text),
            "categories": {
                cat: {
                    "count": len(cat_matches),
                    "max_score": max(m.score for m in cat_matches),
                    "subcategories": list({m.subcategory for m in cat_matches}),
                }
                for cat, cat_matches in by_category.items()
            },
            "detectors_triggered": list({m.detector_name for m in matches}),
        }
