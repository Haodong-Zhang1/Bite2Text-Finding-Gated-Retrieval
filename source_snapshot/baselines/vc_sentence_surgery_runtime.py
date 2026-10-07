#!/usr/bin/env python3
"""Lean full-train sentence-pool runtime for Bite2Text V-C.

This module mirrors the frozen V-C1 sentence mapping and protection rules, but
uses one all-training-reference pool rather than an outer-fold/case lookup.
It intentionally depends only on the rule parser and the Python standard
library so the eventual CPU submission does not need scikit-learn at runtime.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import re
from typing import Any, Mapping

from parsing.schema_parser import FIELDS, hedged, normalize_for_scoring, parse_report


RUNTIME_VERSION = "bite2text_vc_sentence_surgery_full_train_runtime_v1"
COMPACT_POOL_SCHEMA_VERSION = 1

EXPLICIT_TRIGGERS = {
    "dentition": re.compile(
        r"\b(?:early\s+)?mixed dentition\b|\bpermanent dentition\b", re.I
    ),
    "constriction": re.compile(
        r"\bconstriction\b|\btransverse(?:ly)?\b|"
        r"\btransverse (?:relationships?|plane)\b|"
        r"\binter-?arch relationships?\b",
        re.I,
    ),
    "crossbite": re.compile(
        r"\bcross-?bites?\b|\breverse bite\b|\bscissor bite\b", re.I
    ),
    "vertical": re.compile(
        r"\bover-?bite\b|\bdeep ?bite\b|\bopen ?bite\b|"
        r"\bvertical (?:relationships?|standpoint|plane)\b",
        re.I,
    ),
    "overjet": re.compile(r"\bover-?jet\b|\bOVJ\b", re.I),
    "midline": re.compile(r"\bmidlines?\b", re.I),
    "spee": re.compile(r"\bSpee\b", re.I),
    "wilson": re.compile(r"\bWilson\b", re.I),
    "crowding_upper": re.compile(r"\bcrowding\b", re.I),
    "crowding_lower": re.compile(r"\bcrowding\b", re.I),
    "molar_R": re.compile(r"\bmolar\b", re.I),
    "molar_L": re.compile(r"\bmolar\b", re.I),
    "canine_R": re.compile(r"\bcanine\b", re.I),
    "canine_L": re.compile(r"\bcanine\b", re.I),
}

RELATION_FIELDS = frozenset({"molar_R", "molar_L", "canine_R", "canine_L"})
SAFE_ATOMIC_FIELD_SETS = frozenset(
    {
        frozenset({"midline"}),
        frozenset({"spee", "wilson"}),
        frozenset({"crowding_upper", "crowding_lower"}),
        frozenset({"vertical"}),
        frozenset({"overjet"}),
        frozenset({"vertical", "overjet"}),
        frozenset({"constriction"}),
        frozenset({"crossbite"}),
        frozenset({"constriction", "crossbite"}),
        frozenset({"constriction", "vertical"}),
        frozenset({"crossbite", "vertical"}),
        frozenset({"constriction", "crossbite", "vertical"}),
        RELATION_FIELDS,
        RELATION_FIELDS | {"overjet"},
    }
)
EXTRA_CLINICAL_PATTERN = re.compile(
    r"\b(?:car(?:y|ies|ious)|restor(?:ation|ed)|fill(?:ing|ed)|seal(?:ed|ant)|"
    r"gingiv\w*|plaque|pigment\w*|hypoplas\w*|wear|fractur\w*|lesion\w*|"
    r"monitor\w*|diastema\w*|spacing|spaces?|rotat\w*|displac\w*|ectop\w*|"
    r"supernumer\w*|agenes\w*|exfoliat\w*|erupt\w*|deciduous|primary|"
    r"implant\w*|crown\w*|endodont\w*|recession\w*|bridge|splint|torque|"
    r"root\w*|edentulous|tipping|transposition|photo(?:graph)?s?)\b|"
    r"\b(?:weak|full|edge[- ]to[- ]edge|end[- ]to[- ]end|head[- ]to[- ]head|"
    r"super class)\b",
    re.I,
)
CROSSBITE_LOCALIZATION_PATTERN = re.compile(
    r"\b(?:right|left|bilateral|anterior|posterior|lateral|premolar|incisor|"
    r"tooth|teeth)\b|\b(?:molar|canine) region\b",
    re.I,
)
CROWDING_LEVEL_PATTERN = re.compile(
    r"\b(?:no crowding|mild(?:[- ]to[- ]moderate)?|"
    r"moderate(?:[- ]to[- ]severe)?|severe)\b",
    re.I,
)
CROWDING_BOTH_ARCHES_PATTERN = re.compile(
    r"\b(?:upper and lower|lower and upper) arches?\b|\bboth arches\b|"
    r"\bboth the upper and lower arches\b",
    re.I,
)


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    separator: str


@dataclass(frozen=True)
class SentenceSignature:
    fields: tuple[str, ...]
    values: tuple[tuple[str, Any], ...]
    safe: bool
    protect_reason: str | None


@dataclass(frozen=True)
class CompactSentenceCandidate:
    text: str
    frequency: int
    first_source_rank: int


@dataclass(frozen=True)
class CompactSentencePool:
    candidates_by_values: Mapping[
        tuple[tuple[str, Any], ...], tuple[CompactSentenceCandidate, ...]
    ]
    num_reference_reports: int
    num_source_sentences: int
    num_safe_sentences: int


def explicit_sentence_signature(sentence: str) -> SentenceSignature:
    """Mirror the frozen V-C1 explicit mapping and protection policy."""
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    schema = normalize_for_scoring(parse_report(sentence), merge_vertical=True)
    fields = tuple(
        field
        for field in FIELDS
        if schema.get(field) is not None and EXPLICIT_TRIGGERS[field].search(sentence)
    )
    values = tuple((field, schema[field]) for field in fields)
    field_set = frozenset(fields)
    reason: str | None = None
    if not fields:
        reason = "no_explicit_fields"
    elif "dentition" in field_set:
        reason = "protected_dentition"
    elif any(value in {"NA", "unassessable"} for _, value in values):
        reason = "protected_uncertainty"
    elif hedged(sentence):
        reason = "protected_hedged_sentence"
    elif schema.get("missing_teeth") is not None:
        reason = "protected_missing_teeth"
    elif re.search(r"\d", sentence):
        reason = "protected_numeric_content"
    elif EXTRA_CLINICAL_PATTERN.search(sentence):
        reason = "protected_extra_clinical_content"
    elif "crossbite" in field_set and (
        schema.get("crossbite") == "borderline"
        or CROSSBITE_LOCALIZATION_PATTERN.search(sentence)
    ):
        reason = "protected_localized_or_borderline_crossbite"
    elif field_set & {"crowding_upper", "crowding_lower"} and (
        field_set != frozenset({"crowding_upper", "crowding_lower"})
        or schema.get("crowding_upper") != schema.get("crowding_lower")
        or not CROWDING_BOTH_ARCHES_PATTERN.search(sentence)
        or len(CROWDING_LEVEL_PATTERN.findall(sentence)) != 1
        or re.search(r"\b(?:while|whereas|but)\b", sentence, re.I)
    ):
        reason = "unsafe_crowding_surface"
    elif field_set not in SAFE_ATOMIC_FIELD_SETS:
        reason = "unsupported_atomic_field_set"
    return SentenceSignature(fields, values, reason is None, reason)


def split_report_spans(report: str) -> list[SentenceSpan]:
    if not isinstance(report, str):
        raise TypeError("report must be a string")
    if not report:
        return []
    spans: list[SentenceSpan] = []
    start = 0
    index = 0
    while index < len(report):
        boundary = report[index] in ".!?" and (
            index + 1 == len(report) or report[index + 1].isspace()
        )
        if boundary:
            end = index + 1
            separator_end = end
            while separator_end < len(report) and report[separator_end].isspace():
                separator_end += 1
            spans.append(SentenceSpan(report[start:end], report[end:separator_end]))
            start = separator_end
            index = separator_end
            continue
        index += 1
    if start < len(report):
        spans.append(SentenceSpan(report[start:], ""))
    return spans


def render_report_spans(spans: list[SentenceSpan]) -> str:
    if any(not isinstance(span, SentenceSpan) for span in spans):
        raise TypeError("spans must contain SentenceSpan values")
    return "".join(span.text + span.separator for span in spans)


def build_compact_sentence_pool(reference_reports: Mapping[str, str]) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable pool from all train reports."""
    if not reference_reports:
        raise ValueError("reference reports must be non-empty")
    grouped: dict[
        tuple[tuple[str, Any], ...], dict[str, list[int]]
    ] = defaultdict(dict)
    source_rank = 0
    source_sentences = 0
    safe_sentences = 0
    for case_id in sorted(reference_reports):
        report = reference_reports[case_id]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("reference case ids must be non-empty strings")
        if not isinstance(report, str) or not report:
            raise ValueError("every reference report must be a non-empty string")
        for span in split_report_spans(report):
            signature = explicit_sentence_signature(span.text)
            source_sentences += 1
            if signature.safe:
                safe_sentences += 1
                row = grouped[signature.values].setdefault(
                    span.text, [0, source_rank]
                )
                row[0] += 1
                row[1] = min(row[1], source_rank)
            source_rank += 1
    entries = []
    for values, sentences in sorted(grouped.items(), key=lambda item: repr(item[0])):
        candidates = [
            {
                "text": text,
                "frequency": count_and_rank[0],
                "first_source_rank": count_and_rank[1],
            }
            for text, count_and_rank in sorted(
                sentences.items(), key=lambda item: (item[1][1], item[0])
            )
        ]
        entries.append(
            {
                "values": [[field, value] for field, value in values],
                "candidates": candidates,
            }
        )
    return {
        "schema_version": COMPACT_POOL_SCHEMA_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "scope": "all_train_reference_sentences",
        "num_reference_reports": len(reference_reports),
        "num_source_sentences": source_sentences,
        "num_safe_sentences": safe_sentences,
        "entries": entries,
    }


def load_compact_sentence_pool(payload: Mapping[str, Any]) -> CompactSentencePool:
    """Validate and load a compact pool without trusting serialized sentences."""
    expected_keys = {
        "schema_version",
        "runtime_version",
        "scope",
        "num_reference_reports",
        "num_source_sentences",
        "num_safe_sentences",
        "entries",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("compact sentence pool schema differs")
    if (
        payload["schema_version"] != COMPACT_POOL_SCHEMA_VERSION
        or payload["runtime_version"] != RUNTIME_VERSION
        or payload["scope"] != "all_train_reference_sentences"
    ):
        raise ValueError("compact sentence pool version differs")
    counts = [
        payload["num_reference_reports"],
        payload["num_source_sentences"],
        payload["num_safe_sentences"],
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("compact sentence pool counts must be positive integers")
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("compact sentence pool entries must be non-empty")
    result: dict[
        tuple[tuple[str, Any], ...], tuple[CompactSentenceCandidate, ...]
    ] = {}
    observed_safe = 0
    seen_source_ranks: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"values", "candidates"}:
            raise ValueError("compact sentence pool entry is invalid")
        raw_values = entry["values"]
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("compact sentence signature is invalid")
        values = tuple(
            (row[0], row[1])
            for row in raw_values
            if isinstance(row, list) and len(row) == 2
        )
        if len(values) != len(raw_values) or values in result:
            raise ValueError("compact sentence signature differs or is duplicated")
        raw_candidates = entry["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("compact sentence candidates must be non-empty")
        candidates: list[CompactSentenceCandidate] = []
        seen_text: set[str] = set()
        for row in raw_candidates:
            if not isinstance(row, dict) or set(row) != {
                "text",
                "frequency",
                "first_source_rank",
            }:
                raise ValueError("compact sentence candidate is invalid")
            text = row["text"]
            frequency = row["frequency"]
            first_source_rank = row["first_source_rank"]
            if (
                not isinstance(text, str)
                or not text
                or text in seen_text
                or isinstance(frequency, bool)
                or not isinstance(frequency, int)
                or frequency <= 0
                or isinstance(first_source_rank, bool)
                or not isinstance(first_source_rank, int)
                or first_source_rank < 0
            ):
                raise ValueError("compact sentence candidate values are invalid")
            if (
                first_source_rank >= payload["num_source_sentences"]
                or first_source_rank in seen_source_ranks
            ):
                raise ValueError(
                    "compact candidate source rank is invalid or duplicated"
                )
            signature = explicit_sentence_signature(text)
            if not signature.safe or signature.values != values:
                raise ValueError("compact candidate does not match its safe signature")
            seen_text.add(text)
            seen_source_ranks.add(first_source_rank)
            observed_safe += frequency
            candidates.append(
                CompactSentenceCandidate(text, frequency, first_source_rank)
            )
        result[values] = tuple(candidates)
    if observed_safe != payload["num_safe_sentences"]:
        raise ValueError("compact sentence safe-count differs")
    return CompactSentencePool(
        candidates_by_values=result,
        num_reference_reports=payload["num_reference_reports"],
        num_source_sentences=payload["num_source_sentences"],
        num_safe_sentences=payload["num_safe_sentences"],
    )


def _normalized_similarity_text(value: str) -> str:
    return " ".join(value.casefold().split())


def select_sentence_candidate(
    source_sentence: str,
    desired_values: tuple[tuple[str, Any], ...],
    sentence_pool: CompactSentencePool,
) -> tuple[CompactSentenceCandidate, float] | None:
    """Apply the frozen OOF ranking to one compact all-train pool."""
    if not isinstance(sentence_pool, CompactSentencePool):
        raise TypeError("sentence_pool must be a loaded CompactSentencePool")
    candidates = sentence_pool.candidates_by_values.get(desired_values, ())
    if not candidates:
        return None
    source_key = _normalized_similarity_text(source_sentence)

    def rank(candidate: CompactSentenceCandidate) -> tuple[Any, ...]:
        similarity = SequenceMatcher(
            None,
            source_key,
            _normalized_similarity_text(candidate.text),
            autojunk=False,
        ).ratio()
        return (
            -similarity,
            -candidate.frequency,
            abs(len(candidate.text) - len(source_sentence)),
            candidate.first_source_rank,
            candidate.text,
        )

    selected = min(candidates, key=rank)
    similarity = SequenceMatcher(
        None,
        source_key,
        _normalized_similarity_text(selected.text),
        autojunk=False,
    ).ratio()
    return selected, float(similarity)


def _replace_preserving_terminal(source: str, candidate: str) -> str:
    source_terminal = source[-1] if source and source[-1] in ".!?" else ""
    candidate_body = candidate[:-1] if candidate and candidate[-1] in ".!?" else candidate
    return candidate_body + source_terminal


def _normalized_report_schema(report: str) -> dict[str, Any]:
    return normalize_for_scoring(parse_report(report), merge_vertical=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def surgery_report(
    source_report: str,
    target_values: Mapping[str, Any],
    sentence_pool: CompactSentencePool,
) -> tuple[str, dict[str, Any]]:
    """Apply the frozen atomic surgery policy against one full-train pool."""
    if not isinstance(source_report, str) or not source_report:
        raise ValueError("source report must be a non-empty string")
    if not isinstance(target_values, Mapping):
        raise TypeError("target_values must be a mapping")
    if not isinstance(sentence_pool, CompactSentencePool):
        raise TypeError("sentence_pool must be a loaded CompactSentencePool")
    original_spans = split_report_spans(source_report)
    signatures = [explicit_sentence_signature(span.text) for span in original_spans]
    field_locations: dict[str, list[int]] = defaultdict(list)
    for sentence_index, signature in enumerate(signatures):
        for field in signature.fields:
            field_locations[field].append(sentence_index)

    original_schema = _normalized_report_schema(source_report)
    current_spans = list(original_spans)
    replaced_fields: set[str] = set()
    replacements: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    rollback_reasons: Counter[str] = Counter()
    safe_slots_considered = 0

    for sentence_index, signature in enumerate(signatures):
        if not signature.safe:
            skip_reasons[signature.protect_reason or "protected"] += 1
            continue
        safe_slots_considered += 1
        if any(len(field_locations[field]) != 1 for field in signature.fields):
            skip_reasons["nonunique_explicit_field"] += 1
            continue
        if any(
            field not in target_values or target_values[field] is None
            for field in signature.fields
        ):
            skip_reasons["missing_target_value"] += 1
            continue
        desired_values = tuple(
            (field, target_values[field]) for field in signature.fields
        )
        if any(
            value in {"NA", "unassessable", "borderline"}
            for _, value in desired_values
        ):
            skip_reasons["protected_target_uncertainty"] += 1
            continue
        if frozenset(signature.fields) == frozenset(
            {"crowding_upper", "crowding_lower"}
        ) and desired_values[0][1] != desired_values[1][1]:
            skip_reasons["unsafe_target_crowding_pair"] += 1
            continue
        if desired_values == signature.values:
            skip_reasons["already_target"] += 1
            continue
        selection = select_sentence_candidate(
            original_spans[sentence_index].text,
            desired_values,
            sentence_pool,
        )
        if selection is None:
            skip_reasons["no_exact_safe_candidate"] += 1
            continue
        candidate, similarity = selection
        replacement_text = _replace_preserving_terminal(
            original_spans[sentence_index].text,
            candidate.text,
        )
        replacement_signature = explicit_sentence_signature(replacement_text)
        if not replacement_signature.safe or replacement_signature.values != desired_values:
            rollback_reasons["candidate_signature_changed"] += 1
            continue
        proposed_spans = list(current_spans)
        proposed_spans[sentence_index] = SentenceSpan(
            replacement_text,
            original_spans[sentence_index].separator,
        )
        proposed_report = render_report_spans(proposed_spans)
        if len(split_report_spans(proposed_report)) != len(original_spans):
            rollback_reasons["sentence_count_changed"] += 1
            continue
        proposed_schema = _normalized_report_schema(proposed_report)
        proposed_replaced = replaced_fields | set(signature.fields)
        if any(
            proposed_schema.get(field) != target_values[field]
            for field in proposed_replaced
        ):
            rollback_reasons["target_roundtrip_failed"] += 1
            continue
        if any(
            proposed_schema.get(field) != original_schema.get(field)
            for field in FIELDS
            if field not in proposed_replaced
        ):
            rollback_reasons["outside_field_changed"] += 1
            continue
        if proposed_schema.get("missing_teeth") != original_schema.get("missing_teeth"):
            rollback_reasons["missing_teeth_changed"] += 1
            continue
        current_spans = proposed_spans
        replaced_fields = proposed_replaced
        replacements.append(
            {
                "sentence_index": sentence_index,
                "fields": list(signature.fields),
                "desired_values": dict(desired_values),
                "source_sentence_sha256": _sha256_text(
                    original_spans[sentence_index].text
                ),
                "candidate_sentence_sha256": _sha256_text(candidate.text),
                "candidate_frequency": candidate.frequency,
                "candidate_first_source_rank": candidate.first_source_rank,
                "sequence_matcher_similarity": similarity,
            }
        )

    output_report = render_report_spans(current_spans)
    return output_report, {
        "status": "changed" if replacements else "unchanged",
        "source_report_sha256": _sha256_text(source_report),
        "output_report_sha256": _sha256_text(output_report),
        "source_sentence_count": len(original_spans),
        "output_sentence_count": len(split_report_spans(output_report)),
        "safe_slots_considered": safe_slots_considered,
        "replacements": replacements,
        "skip_reason_counts": dict(sorted(skip_reasons.items())),
        "rollback_reason_counts": dict(sorted(rollback_reasons.items())),
    }
