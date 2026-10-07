"""High-confidence gate for unsupported tooth-level donor details."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Mapping, Sequence

from baselines.vc_sentence_surgery_runtime import (
    SentenceSpan,
    render_report_spans,
    split_report_spans,
)


RUNTIME_VERSION = "bitevlm_lite_tooth_finding_gate_v1"
MIN_NEIGHBOR_SUPPORT = 0.80

_NEGATION = re.compile(
    r"\b(?:no|not|without|absence\s+of|free\s+of|negative\s+for)\b|"
    r"\b(?:do|does|did)\s+not\b|\bcannot\b",
    re.I,
)
_CORE_RELATION = re.compile(
    r"\b(?:dentition|constriction|transverse(?:ly)?|vertical|sagitt(?:al|ally)|"
    r"over-?bite|over-?jet|midlines?|spee|wilson|crowding|cross-?bite|"
    r"molar\s+(?:and\s+)?canine\s+relationship)\b",
    re.I,
)

FAMILY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "restoration": re.compile(
        r"\b(?:restor(?:ation|ative|ed)\w*|fill(?:ing|ed)\w*|"
        r"prosthetic\s+crown\w*|crown(?:s)?)\b",
        re.I,
    ),
    "caries": re.compile(r"\b(?:caries|carious|cavity|decay)\b", re.I),
    "sealant": re.compile(
        r"\b(?:sealant(?:s)?|sealed|fissure\s+seal\w*)\b", re.I
    ),
    "demineralisation": re.compile(
        r"\b(?:deminerali[sz]\w*|white[- ]spot(?:s)?)\b", re.I
    ),
    "supernumerary": re.compile(r"\bsupernumerary\b", re.I),
    "microdontia": re.compile(r"\bmicrodont\w*\b", re.I),
    "recession": re.compile(r"\brecession(?:s)?\b", re.I),
}

GENERIC_FINDINGS: Mapping[str, str] = {
    "restoration": "Dental restorations are present",
    "caries": "A carious lesion is present",
    "sealant": "Pit and fissure sealant material is present",
    "demineralisation": "Enamel demineralisation is present",
    "supernumerary": "A supernumerary tooth is present",
    "microdontia": "Microdontia is present",
    "recession": "Gingival recession is present",
}

_FDI = r"(?:[1-4][1-8]|[5-8][1-5])"
_FDI_LIST = rf"{_FDI}(?:\s*(?:,|/|[-–]|\band\b|&)\s*{_FDI})*"
_CROSSBITE_TERM = r"(?:cross-?bite|reverse\s+bite|scissor\s+bite)"
_CROSSBITE_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)(?:"
    rf"\s+(?:involving|affecting)\s+(?:(?:tooth|teeth)\s+)?"
    rf"(?:(?:the\s+)?(?:entire|whole|full)\s+"
    rf"(?:(?:right\s+and\s+left)\s+hemi[- ]?)?arch"
    rf"(?:\s+except\s+for\s+(?:(?:tooth|teeth)\s+)?{_FDI_LIST})?"
    rf"|{_FDI_LIST})"
    rf"|\s+(?:on|at|between)\s+"
    rf"(?:(?:the\s+)?(?:level\s+of\s+)?|(?:site|element)\s+)"
    rf"(?:(?:tooth|teeth)\s+)?{_FDI_LIST}"
    rf")",
    re.I,
)
_CROSSBITE_NAMED_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+(?:on|at|in)\s+(?:the\s+)?"
    rf"(?:right|left|bilateral)(?:[- ]sided|\s+side)?"
    rf"(?:"
    rf"\s+(?:at|in)\s+(?:the\s+)?(?:level\s+of\s+)?"
    rf"(?:(?:tooth|teeth)\s+)?{_FDI_LIST}"
    rf"|\s+(?:at|in)\s+(?:the\s+)?"
    rf"(?:canine|premolar|molar|incisor|posterior|anterior|lateral)"
    rf"(?:\s*[-–]\s*(?:canine|premolar|molar|incisor))?"
    rf"(?:\s*,\s*(?:premolar|molar|canine|incisor))*"
    rf"(?:\s*,?\s+and\s+(?:premolar|molar|canine|incisor))?"
    rf"\s+regions?"
    rf")?",
    re.I,
)
_CROSSBITE_REGION_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+(?:at|in)\s+(?:the\s+)?(?:level\s+of\s+)?"
    rf"(?:the\s+)?(?:canine|premolar|molar|incisor|posterior|anterior|lateral)"
    rf"(?:\s*[-–]\s*(?:canine|premolar|molar|incisor))?\s+regions?",
    re.I,
)
_CROSSBITE_NUMERIC_REGION_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+(?:at|in)\s+(?:the\s+)?"
    rf"(?:(?:area|region)\s+of\s+{_FDI_LIST}|{_FDI_LIST}\s+regions?)",
    re.I,
)
_CROSSBITE_NUMERIC_PARENTHESIS = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s*\([^)]*\b{_FDI}\b[^)]*\)",
    re.I,
)
_CROSSBITE_GENERIC_TOOTH_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+(?:on|at)\s+(?:the\s+)?(?:tooth|teeth|incisors?)\b",
    re.I,
)
_CROSSBITE_ALONG_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+(?:along|across)\s+(?:the\s+)?"
    rf"(?:(?:entire|whole|full)\s+)?(?:right|left|bilateral)?\s*side"
    rf"(?:\s+in\s+(?:the\s+)?(?:canine|premolar|molar|incisor)"
    rf"(?:\s*,\s*(?:premolar|molar|canine|incisor))*"
    rf"(?:\s*,?\s+and\s+(?:premolar|molar|canine|incisor))?"
    rf"\s+regions?)?",
    re.I,
)
_CROSSBITE_EXTENSION_SCOPE = re.compile(
    rf"(?P<prefix>\b{_CROSSBITE_TERM}\b"
    rf"(?:\s+is\s+(?:present|noted|observed|evident))?)"
    rf"\s+extending\s+from\s+[^.!?]+",
    re.I,
)
_CROSSBITE_LOCATION_PREFIX = re.compile(
    rf"(?P<article>\b(?:a|an)\s+)?"
    rf"(?:(?:right|left|bilateral|anterior|posterior|lateral|"
    rf"right[- ]sided|left[- ]sided)\s+)+"
    rf"(?P<term>{_CROSSBITE_TERM})\b",
    re.I,
)
_CROSSBITE_LEADING_SITE = re.compile(
    rf"\b(?:(?:tooth|teeth)\s+)?{_FDI_LIST}\s+(?:is|are)\s+in\s+"
    rf"(?P<qualifier>(?:(?:very\s+)?(?:slight|mild)\s+)?)"
    rf"(?P<term>{_CROSSBITE_TERM})\b",
    re.I,
)
_SITE_LED_CROSSBITE_FINAL = re.compile(
    rf"^(?P<prefix>.*?)\s*;\s*however,\s*"
    rf"at\s+(?:site|tooth)\s+{_FDI}\b[^.!?]*\b{_CROSSBITE_TERM}\b"
    rf"[^.!?]*(?P<terminal>[.!?]?)$",
    re.I,
)
_EXCEPT_SITE_CROSSBITE_BEFORE_WHILE = re.compile(
    rf",\s*except\s+at\s+(?:site|tooth)\s+{_FDI}\b[^.!?]*?"
    rf"\b{_CROSSBITE_TERM}\b[^.!?]*?(?=,\s*while\b)",
    re.I,
)
_EXCEPT_SITE_CROSSBITE_FINAL = re.compile(
    rf",\s*except\s+at\s+(?:site|tooth)\s+{_FDI}\b[^.!?]*?"
    rf"\b{_CROSSBITE_TERM}\b[^.!?]*(?=[.!?]?$)",
    re.I,
)
_PARENTHETICAL_SITE_CROSSBITE = re.compile(
    rf"\s*\((?:in\s+particular\s+)?(?:site|tooth)\s+{_FDI}\s*,?"
    rf"\s+which\b[^)]*\b{_CROSSBITE_TERM}\b[^)]*\)",
    re.I,
)
_CONSEQUENT_CROSSBITE = re.compile(
    rf"\s+with\s+consequent\s+(?:(?:anterior|posterior|lateral)\s+)?"
    rf"{_CROSSBITE_TERM}\b",
    re.I,
)
_COMPOSITE_DENTAL_CROSSBITE_CLAUSE = re.compile(
    rf";\s*however,\s*there\s+(?:are|is)\s+(?:a\s+)?dental\s+"
    rf"cross-?bites?\s+with\s+(?:tooth|teeth)\b[^.!?]*(?=[.!?]?$)",
    re.I,
)
_TRAILING_SITE_PATHOLOGY_CLAUSES = (
    re.compile(
        r"^(?P<prefix>.+?);\s*however,\s*(?P<clause>.+?)"
        r"(?P<terminal>[.!?]?)$",
        re.I,
    ),
    re.compile(
        r"^(?P<prefix>.+?),\s*with\s+(?P<clause>.+?)"
        r"(?P<terminal>[.!?]?)$",
        re.I,
    ),
)


@dataclass(frozen=True)
class GateAudit:
    changed: bool
    crossbite_scopes_removed: int
    pathology_sentences_removed: int
    pathology_sentences_minimised: int
    supported_families: tuple[str, ...]


def _finite_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def positive_families(text: str) -> frozenset[str]:
    """Return targeted positive finding families mentioned in ``text``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    families: set[str] = set()
    for span in split_report_spans(text):
        if _NEGATION.search(span.text):
            continue
        families.update(
            name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(span.text)
        )
    return frozenset(families)


def neighbor_support(reports: Sequence[str]) -> dict[str, float]:
    """Estimate case-local family support from the fixed retrieval shortlist."""
    if not reports or any(not isinstance(report, str) for report in reports):
        raise ValueError("neighbor reports must be a non-empty text sequence")
    counts = {name: 0 for name in FAMILY_PATTERNS}
    for report in reports:
        for family in positive_families(report):
            counts[family] += 1
    return {name: count / len(reports) for name, count in counts.items()}


def _generic_sentence(families: Sequence[str], punctuation: str) -> str:
    statements = [GENERIC_FINDINGS[name] for name in families]
    if not statements:
        return ""
    ending = punctuation if punctuation in ".!?" else "."
    return ". ".join(statements) + ending


def _gate_trailing_site_pathology_clause(
    sentence: str,
    probabilities: Mapping[str, float],
    threshold: float,
) -> tuple[str, int, int, tuple[str, ...]]:
    for pattern in _TRAILING_SITE_PATHOLOGY_CLAUSES:
        match = pattern.match(sentence)
        if match is None:
            continue
        clause = match.group("clause")
        if not re.search(rf"\b{_FDI}\b", clause) or _NEGATION.search(clause):
            continue
        families = tuple(
            name for name, family_pattern in FAMILY_PATTERNS.items()
            if family_pattern.search(clause)
        )
        if len(families) != 1:
            continue
        terminal = match.group("terminal") or "."
        prefix = match.group("prefix").rstrip() + terminal
        family = families[0]
        if probabilities[family] >= threshold:
            generic = _generic_sentence((family,), terminal)
            return f"{prefix} {generic}", 0, 1, (family,)
        return prefix, 1, 0, ()
    return sentence, 0, 0, ()


def _remove_crossbite_localization(sentence: str) -> tuple[str, int]:
    edits = 0
    for _ in range(3):
        round_edits = 0
        for pattern in (
            _CROSSBITE_SCOPE,
            _CROSSBITE_NAMED_SCOPE,
            _CROSSBITE_REGION_SCOPE,
            _CROSSBITE_NUMERIC_REGION_SCOPE,
            _CROSSBITE_NUMERIC_PARENTHESIS,
            _CROSSBITE_GENERIC_TOOTH_SCOPE,
            _CROSSBITE_ALONG_SCOPE,
            _CROSSBITE_EXTENSION_SCOPE,
        ):
            sentence, count = pattern.subn(
                lambda match: match.group("prefix"), sentence
            )
            round_edits += count

        def replace_prefix(match: re.Match[str]) -> str:
            term = match.group("term")
            article = match.group("article")
            if not article:
                return term
            replacement_article = "A" if article[0].isupper() else "a"
            return f"{replacement_article} {term}"

        sentence, count = _CROSSBITE_LOCATION_PREFIX.subn(replace_prefix, sentence)
        round_edits += count

        def replace_leading_site(match: re.Match[str]) -> str:
            qualifier = match.group("qualifier") or ""
            article = "A" if match.group(0)[0].isupper() else "a"
            return f"{article} {qualifier}{match.group('term')} is present"

        sentence, count = _CROSSBITE_LEADING_SITE.subn(
            replace_leading_site, sentence
        )
        round_edits += count
        edits += round_edits
        if not round_edits:
            break

    match = _SITE_LED_CROSSBITE_FINAL.match(sentence)
    if match is not None:
        sentence = match.group("prefix").rstrip() + (match.group("terminal") or ".")
        edits += 1
    sentence, count = _EXCEPT_SITE_CROSSBITE_BEFORE_WHILE.subn("", sentence)
    edits += count
    sentence, count = _EXCEPT_SITE_CROSSBITE_FINAL.subn("", sentence)
    edits += count
    sentence, count = _PARENTHETICAL_SITE_CROSSBITE.subn("", sentence)
    edits += count
    sentence, count = _CONSEQUENT_CROSSBITE.subn("", sentence)
    edits += count
    sentence, count = _COMPOSITE_DENTAL_CROSSBITE_CLAUSE.subn(
        "; however, there is a crossbite", sentence
    )
    edits += count
    return sentence, edits


def remove_crossbite_localization(report: str) -> tuple[str, int]:
    """Remove crossbite site/range wording while preserving report structure."""
    if not isinstance(report, str) or not report:
        raise ValueError("report must be non-empty text")
    spans: list[SentenceSpan] = []
    edits = 0
    for span in split_report_spans(report):
        sentence, count = _remove_crossbite_localization(span.text)
        spans.append(SentenceSpan(sentence, span.separator))
        edits += count
    return render_report_spans(spans), edits


def gate_report(
    report: str,
    *,
    support: Mapping[str, float],
    min_support: float = MIN_NEIGHBOR_SUPPORT,
) -> tuple[str, GateAudit]:
    """Remove unsupported pathology detail and crossbite tooth scope."""
    if not isinstance(report, str) or not report:
        raise ValueError("report must be non-empty text")
    threshold = _finite_probability(min_support)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_support must be finite and within [0, 1]")
    probabilities: dict[str, float] = {}
    for family in FAMILY_PATTERNS:
        probability = _finite_probability(support.get(family))
        if probability is None or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid support for {family}")
        probabilities[family] = probability

    kept: list[SentenceSpan] = []
    scopes_removed = 0
    removed = 0
    minimised = 0
    supported: set[str] = set()
    for span in split_report_spans(report):
        sentence, clause_removed, clause_minimised, clause_supported = (
            _gate_trailing_site_pathology_clause(
                span.text, probabilities, threshold
            )
        )
        removed += clause_removed
        minimised += clause_minimised
        supported.update(clause_supported)
        sentence, scope_count = _remove_crossbite_localization(sentence)
        scopes_removed += scope_count
        families = tuple(
            name
            for name, pattern in FAMILY_PATTERNS.items()
            if pattern.search(sentence) and not _NEGATION.search(sentence)
        )
        if families and not _CORE_RELATION.search(sentence):
            retained = tuple(
                family
                for family in families
                if probabilities[family] >= threshold
            )
            supported.update(retained)
            if not retained:
                removed += 1
                continue
            punctuation = sentence[-1:] if sentence else ""
            sentence = _generic_sentence(retained, punctuation)
            minimised += 1
        kept.append(SentenceSpan(sentence, span.separator))

    gated = render_report_spans(kept).strip()
    if not gated:
        return report, GateAudit(False, 0, 0, 0, ())
    audit = GateAudit(
        changed=gated != report,
        crossbite_scopes_removed=scopes_removed,
        pathology_sentences_removed=removed,
        pathology_sentences_minimised=minimised,
        supported_families=tuple(sorted(supported)),
    )
    return gated, audit
