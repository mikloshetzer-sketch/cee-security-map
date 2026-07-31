#!/usr/bin/env python3
"""
CEE Security Map - Security Classifier v1.0

Loads the machine-readable security taxonomy from the repository root and
classifies article text into a structured SecurityIncident object.

Expected repository structure:

    security_taxonomy.json
    scripts/
        security_classifier.py

The module is intentionally independent from RSS fetching and output
enrichment. It only performs classification.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("security_classifier")
CLASSIFIER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "security_taxonomy.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SecurityIncident:
    """Canonical classifier output consumed by the rest of the pipeline."""

    id: str
    title: str
    summary: str
    source: str
    url: str
    published: str | None

    country: str | None
    city: str | None
    latitude: float | None
    longitude: float | None

    family: str
    subcategory: str
    subtype: str

    object: str
    action: str
    actor: str
    target: str
    context: list[str]
    consequences: list[str]

    severity: str
    confidence: float
    article_role: str
    actual_incident: bool

    matched_rule_id: str | None
    fingerprint: dict[str, Any]

    classifier_version: str = CLASSIFIER_VERSION
    taxonomy_version: str = ""
    matched_terms: dict[str, list[str]] = field(default_factory=dict)
    negative_terms: list[str] = field(default_factory=list)
    confidence_components: dict[str, float] = field(default_factory=dict)
    candidate_rules: list[dict[str, Any]] = field(default_factory=list)
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_geojson_properties(self) -> dict[str, Any]:
        """Return a GeoJSON-friendly representation."""
        data = self.to_dict()
        data.pop("latitude", None)
        data.pop("longitude", None)
        return data


@dataclass(slots=True)
class RuleCandidate:
    rule_id: str
    family: str
    subcategory: str
    subtype: str
    severity: str
    score: float
    matched: bool
    missing: list[str] = field(default_factory=list)
    matched_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None

    candidates = (
        text,
        text.replace("Z", "+00:00"),
    )
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue
    return text


def _date_bucket(value: str | None) -> str:
    if not value:
        return "unknown-date"
    return value[:10] if len(value) >= 10 else value


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class SecurityClassifier:
    """Taxonomy-driven rule engine."""

    def __init__(
        self,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
        *,
        validate: bool = True,
    ) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy(self.taxonomy_path)

        if validate:
            self._validate_taxonomy(self.taxonomy)

        self.metadata = self.taxonomy["metadata"]
        self.runtime = self.taxonomy.get("runtime", {})
        self.lexicons = self.taxonomy.get("lexicons", {})
        self.article_roles = self.taxonomy.get("article_roles", {})
        self.composition_rules = self.taxonomy.get("composition_rules", [])
        self.negative_contexts = self.taxonomy.get("negative_contexts", {})
        self.severity_model = self.taxonomy.get("severity_model", {})
        self.confidence_model = self.taxonomy.get("confidence_model", {})
        self.fingerprint_model = self.taxonomy.get("fingerprint", {})
        self.output_schema = self.taxonomy.get("output_schema", {})
        self.debug_config = self.taxonomy.get("debug", {})

        self._compiled_lexicons = self._compile_lexicons(self.lexicons)
        self._compiled_roles = self._compile_article_roles(self.article_roles)
        self._compiled_negative_contexts = self._compile_negative_contexts(
            self.negative_contexts
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        *,
        title: str,
        summary: str = "",
        source: str = "",
        url: str = "",
        published: str | None = None,
        country: str | None = None,
        city: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        source_is_official: bool = False,
        source_count: int = 1,
        extra_context: Mapping[str, Any] | None = None,
    ) -> SecurityIncident:
        """
        Classify one article.

        Country and city detection remain external responsibilities of
        fetch_local_sources.py. The classifier consumes them as evidence.
        """
        normalized_title = self.normalize_text(title)
        normalized_summary = self.normalize_text(summary)
        text = f"{normalized_title} {normalized_summary}".strip()

        role, role_matches = self.detect_article_role(text)
        negative_terms = self.detect_negative_contexts(text)

        matched = {
            "objects": self.detect_lexicon_group("objects", text),
            "actions": self.detect_lexicon_group("actions", text),
            "targets": self.detect_lexicon_group("targets", text),
            "actors": self.detect_lexicon_group("actors", text),
            "consequences": self.detect_lexicon_group("consequences", text),
        }

        candidates = self.apply_composition_rules(text=text, matched=matched)
        best_candidate = self.select_best_candidate(candidates)

        family = best_candidate.family if best_candidate else "unknown"
        subcategory = best_candidate.subcategory if best_candidate else "unknown"
        subtype = best_candidate.subtype if best_candidate else "unknown"
        matched_rule_id = best_candidate.rule_id if best_candidate else None

        primary_object = self._first_key(matched["objects"])
        primary_action = self._first_key(matched["actions"])
        primary_actor = self._first_key(matched["actors"])
        primary_target = self._first_key(matched["targets"])

        context = self._collect_context_terms(candidates)
        consequences = list(matched["consequences"].keys())

        confidence, confidence_components = self.calculate_confidence(
            family=family,
            article_role=role,
            matched=matched,
            country=country,
            city=city,
            source_is_official=source_is_official,
            source_count=source_count,
            negative_terms=negative_terms,
            candidate=best_candidate,
        )

        severity = self.calculate_severity(
            family=family,
            base_rule_severity=best_candidate.severity if best_candidate else None,
            article_role=role,
            action=primary_action,
            target=primary_target,
            consequences=consequences,
            source_count=source_count,
        )

        actual_incident, rejection_reason = self.determine_actual_incident(
            article_role=role,
            confidence=confidence,
            matched_rule_id=matched_rule_id,
            negative_terms=negative_terms,
        )

        published_iso = _iso_date(published)
        fingerprint = self.generate_fingerprint(
            country=country,
            family=family,
            subcategory=subcategory,
            subtype=subtype,
            object_name=primary_object,
            action=primary_action,
            actor=primary_actor,
            target=primary_target,
            place=city,
            published=published_iso,
        )

        incident_id = self.generate_incident_id(
            url=url,
            title=title,
            published=published_iso,
            fingerprint=fingerprint,
        )

        debug_matches = {
            "article_role": role_matches,
            "objects": self._flatten_matches(matched["objects"]),
            "actions": self._flatten_matches(matched["actions"]),
            "targets": self._flatten_matches(matched["targets"]),
            "actors": self._flatten_matches(matched["actors"]),
            "consequences": self._flatten_matches(matched["consequences"]),
        }

        if extra_context:
            debug_matches["extra_context"] = [
                f"{key}={value}" for key, value in extra_context.items()
            ]

        return SecurityIncident(
            id=incident_id,
            title=title.strip(),
            summary=summary.strip(),
            source=source.strip(),
            url=url.strip(),
            published=published_iso,
            country=country,
            city=city,
            latitude=_safe_float(latitude),
            longitude=_safe_float(longitude),
            family=family,
            subcategory=subcategory,
            subtype=subtype,
            object=primary_object,
            action=primary_action,
            actor=primary_actor,
            target=primary_target,
            context=context,
            consequences=consequences,
            severity=severity,
            confidence=round(confidence, 4),
            article_role=role,
            actual_incident=actual_incident,
            matched_rule_id=matched_rule_id,
            fingerprint=fingerprint,
            taxonomy_version=str(self.metadata.get("version", "")),
            matched_terms=debug_matches,
            negative_terms=negative_terms,
            confidence_components={
                key: round(value, 4)
                for key, value in confidence_components.items()
            },
            candidate_rules=[candidate.to_dict() for candidate in candidates[:10]],
            rejection_reason=rejection_reason,
        )

    def classify_article(self, article: Mapping[str, Any]) -> SecurityIncident:
        """Convenience wrapper for dictionary-based pipeline records."""
        return self.classify(
            title=str(article.get("title", "")),
            summary=str(
                article.get("summary")
                or article.get("description")
                or article.get("content")
                or ""
            ),
            source=str(article.get("source") or article.get("source_name") or ""),
            url=str(article.get("url") or article.get("link") or ""),
            published=article.get("published") or article.get("published_at"),
            country=article.get("country"),
            city=article.get("city"),
            latitude=article.get("latitude") or article.get("lat"),
            longitude=article.get("longitude") or article.get("lon"),
            source_is_official=bool(article.get("source_is_official", False)),
            source_count=int(article.get("source_count", 1) or 1),
        )

    # ------------------------------------------------------------------
    # Taxonomy loading and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _load_taxonomy(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid taxonomy JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise TypeError("Taxonomy root must be a JSON object.")
        return data

    @staticmethod
    def _validate_taxonomy(data: Mapping[str, Any]) -> None:
        required_sections = {
            "metadata",
            "families",
            "lexicons",
            "composition_rules",
            "confidence_model",
            "severity_model",
            "output_schema",
        }
        missing = sorted(required_sections - set(data))
        if missing:
            raise ValueError(
                "Taxonomy is missing required sections: " + ", ".join(missing)
            )

        lexicons = data.get("lexicons", {})
        for group in ("objects", "actions", "targets", "actors", "consequences"):
            if group not in lexicons or not isinstance(lexicons[group], dict):
                raise ValueError(f"Taxonomy lexicon group is missing: {group}")

        known_families = set(data.get("families", {}))
        for rule in data.get("composition_rules", []):
            if not isinstance(rule, dict):
                raise TypeError("Each composition rule must be an object.")
            for key in ("rule_id", "family", "subcategory", "subtype"):
                if not rule.get(key):
                    raise ValueError(f"Composition rule missing '{key}': {rule}")
            if rule["family"] not in known_families:
                raise ValueError(
                    f"Unknown family '{rule['family']}' in {rule['rule_id']}"
                )

    # ------------------------------------------------------------------
    # Text handling
    # ------------------------------------------------------------------

    def normalize_text(self, value: str | None) -> str:
        if not value:
            return ""

        config = self.runtime.get("normalization", {})
        text = html.unescape(str(value))
        text = re.sub(r"<[^>]+>", " ", text)

        normalization_form = config.get("unicode_normalization", "NFKC")
        text = unicodedata.normalize(normalization_form, text)

        if config.get("lowercase", True):
            text = text.casefold()

        if config.get("collapse_whitespace", True):
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _term_pattern(term: str) -> re.Pattern[str]:
        escaped = re.escape(term.casefold())
        # Unicode-aware boundaries without relying on \b for punctuation.
        pattern = rf"(?<!\w){escaped}(?!\w)"
        return re.compile(pattern, flags=re.IGNORECASE)

    def _compile_lexicons(
        self,
        lexicons: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> dict[str, dict[str, list[tuple[str, re.Pattern[str]]]]]:
        result: dict[str, dict[str, list[tuple[str, re.Pattern[str]]]]] = {}
        for group_name, entries in lexicons.items():
            group_result: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
            for canonical, terms in entries.items():
                compiled_terms: list[tuple[str, re.Pattern[str]]] = []
                for term in terms:
                    normalized = self.normalize_text(term)
                    if normalized:
                        compiled_terms.append(
                            (normalized, self._term_pattern(normalized))
                        )
                group_result[canonical] = compiled_terms
            result[group_name] = group_result
        return result

    def _compile_article_roles(
        self,
        roles: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        result: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for role, config in roles.items():
            compiled: list[tuple[str, re.Pattern[str]]] = []
            for term in config.get("positive_terms", []):
                normalized = self.normalize_text(term)
                if normalized:
                    compiled.append((normalized, self._term_pattern(normalized)))
            result[role] = compiled
        return result

    def _compile_negative_contexts(
        self,
        groups: Mapping[str, Sequence[str]],
    ) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        result: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for group, terms in groups.items():
            result[group] = []
            for term in terms:
                normalized = self.normalize_text(term)
                if normalized:
                    result[group].append(
                        (normalized, self._term_pattern(normalized))
                    )
        return result

    # ------------------------------------------------------------------
    # Entity detection
    # ------------------------------------------------------------------

    def detect_lexicon_group(
        self,
        group_name: str,
        text: str,
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for canonical, compiled_terms in self._compiled_lexicons.get(
            group_name, {}
        ).items():
            matches = [
                term for term, pattern in compiled_terms if pattern.search(text)
            ]
            if matches:
                result[canonical] = _stable_unique(matches)
        return result

    def detect_article_role(self, text: str) -> tuple[str, list[str]]:
        matches_by_role: dict[str, list[str]] = {}
        for role, compiled_terms in self._compiled_roles.items():
            matches = [
                term for term, pattern in compiled_terms if pattern.search(text)
            ]
            if matches:
                matches_by_role[role] = _stable_unique(matches)

        # Explicitly non-incident roles take precedence over generic incident
        # language, because reaction and analysis articles often mention the
        # original incident in the same title or summary.
        priority = [
            "procurement",
            "exercise",
            "historical",
            "analysis",
            "summary",
            "reaction",
            "followup",
            "incident",
        ]
        for role in priority:
            if role in matches_by_role:
                return role, matches_by_role[role]

        return "incident", []

    def detect_negative_contexts(self, text: str) -> list[str]:
        matches: list[str] = []
        for compiled_terms in self._compiled_negative_contexts.values():
            for term, pattern in compiled_terms:
                if pattern.search(text):
                    matches.append(term)
        return _stable_unique(matches)

    # ------------------------------------------------------------------
    # Composition rules
    # ------------------------------------------------------------------

    def apply_composition_rules(
        self,
        *,
        text: str,
        matched: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> list[RuleCandidate]:
        candidates: list[RuleCandidate] = []

        for rule in self.composition_rules:
            missing: list[str] = []
            matched_context: list[str] = []
            score = 0.0

            requires = rule.get("requires", {})

            for group_key, required_values in requires.items():
                canonical_group, mode = self._parse_requirement_key(group_key)
                available = set(matched.get(canonical_group, {}).keys())
                required = set(required_values)

                if mode == "all":
                    absent = sorted(required - available)
                    if absent:
                        missing.append(
                            f"{canonical_group}:all:{','.join(absent)}"
                        )
                    else:
                        score += 1.0 + 0.1 * len(required)
                else:
                    overlap = required & available
                    if not overlap:
                        missing.append(
                            f"{canonical_group}:any:{','.join(sorted(required))}"
                        )
                    else:
                        score += 1.0 + 0.1 * len(overlap)

            context_terms = [
                self.normalize_text(item)
                for item in rule.get("context_any", [])
                if item
            ]
            if context_terms:
                matched_context = [
                    term
                    for term in context_terms
                    if self._term_pattern(term).search(text)
                ]
                if not matched_context:
                    missing.append("context:any")
                else:
                    score += 0.8 + min(0.2, len(matched_context) * 0.05)

            matched_rule = not missing

            # More specific rules win when several candidates match.
            specificity = (
                sum(len(v) for v in requires.values())
                + len(context_terms)
            )
            if matched_rule:
                score += min(0.5, specificity * 0.03)

            candidates.append(
                RuleCandidate(
                    rule_id=str(rule["rule_id"]),
                    family=str(rule["family"]),
                    subcategory=str(rule["subcategory"]),
                    subtype=str(rule["subtype"]),
                    severity=str(
                        rule.get("severity")
                        or self.taxonomy["families"]
                        .get(rule["family"], {})
                        .get("default_severity", "medium")
                    ),
                    score=round(score, 4),
                    matched=matched_rule,
                    missing=missing,
                    matched_context=matched_context,
                )
            )

        candidates.sort(
            key=lambda item: (item.matched, item.score),
            reverse=True,
        )
        return candidates

    @staticmethod
    def _parse_requirement_key(key: str) -> tuple[str, str]:
        if key.endswith("_any"):
            return key[:-4], "any"
        return key, "all"

    @staticmethod
    def select_best_candidate(
        candidates: Sequence[RuleCandidate],
    ) -> RuleCandidate | None:
        for candidate in candidates:
            if candidate.matched:
                return candidate
        return None

    @staticmethod
    def _collect_context_terms(
        candidates: Sequence[RuleCandidate],
    ) -> list[str]:
        values: list[str] = []
        for candidate in candidates:
            if candidate.matched:
                values.extend(candidate.matched_context)
        return _stable_unique(values)

    # ------------------------------------------------------------------
    # Severity and confidence
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        *,
        family: str,
        article_role: str,
        matched: Mapping[str, Mapping[str, Sequence[str]]],
        country: str | None,
        city: str | None,
        source_is_official: bool,
        source_count: int,
        negative_terms: Sequence[str],
        candidate: RuleCandidate | None,
    ) -> tuple[float, dict[str, float]]:
        model = self.confidence_model
        weights = model.get("weights", {})
        penalties = model.get("penalties", {})

        value = float(model.get("base", 0.20))
        components: dict[str, float] = {"base": value}

        def add_component(name: str, condition: bool) -> None:
            nonlocal value
            if condition:
                amount = float(weights.get(name, 0.0))
                value += amount
                components[name] = amount

        add_component("object_match", bool(matched["objects"]))
        add_component("action_match", bool(matched["actions"]))
        add_component("target_match", bool(matched["targets"]))
        add_component("actor_match", bool(matched["actors"]))
        add_component("country_match", bool(country))
        add_component("city_match", bool(city))
        add_component("consequence_match", bool(matched["consequences"]))
        add_component("official_source_signal", source_is_official)
        add_component("multiple_source_signal", source_count >= 2)
        add_component("supporting_evidence", candidate is not None)

        def subtract_component(name: str, condition: bool) -> None:
            nonlocal value
            if condition:
                amount = float(penalties.get(name, 0.0))
                value -= amount
                components[f"penalty:{name}"] = -amount

        subtract_component("negative_context", bool(negative_terms))
        subtract_component(
            "article_role_non_incident",
            article_role
            in {
                "reaction",
                "analysis",
                "summary",
                "historical",
                "procurement",
                "exercise",
            },
        )
        subtract_component("missing_action", not matched["actions"])
        subtract_component("missing_object", not matched["objects"])

        caps = model.get("caps", {})
        minimum = float(caps.get("minimum", 0.0))
        maximum = float(caps.get("maximum", 0.99))
        value = max(minimum, min(maximum, value))

        # A valid composition rule is mandatory for high confidence.
        if candidate is None:
            value = min(value, 0.67)
            components["cap:no_composition_rule"] = value

        return value, components

    def calculate_severity(
        self,
        *,
        family: str,
        base_rule_severity: str | None,
        article_role: str,
        action: str,
        target: str,
        consequences: Sequence[str],
        source_count: int,
    ) -> str:
        levels = self.severity_model.get(
            "levels",
            ["info", "low", "medium", "high", "critical"],
        )
        level_index = {name: index for index, name in enumerate(levels)}

        severity = (
            base_rule_severity
            or self.severity_model.get("base_by_family", {}).get(family)
            or "info"
        )

        def at_least(current: str, minimum: str) -> str:
            if level_index.get(current, 0) >= level_index.get(minimum, 0):
                return current
            return minimum

        def at_most(current: str, maximum: str) -> str:
            if level_index.get(current, 0) <= level_index.get(maximum, 0):
                return current
            return maximum

        for rule in self.severity_model.get("upgrade_rules", []):
            consequence = rule.get("when_consequence")
            if consequence and consequence not in consequences:
                continue

            when_target = rule.get("when_target")
            if when_target and when_target != target:
                continue

            actions = rule.get("and_action_any")
            if actions and action not in actions:
                continue

            source_threshold = rule.get("when_multiple_sources_at_least")
            if source_threshold and source_count < int(source_threshold):
                continue

            if rule.get("confidence_only"):
                continue

            if rule.get("minimum"):
                severity = at_least(severity, str(rule["minimum"]))

        for rule in self.severity_model.get("downgrade_rules", []):
            roles = rule.get("when_article_role")
            if roles and article_role not in roles:
                continue

            when_action = rule.get("when_action")
            if when_action and action != when_action:
                continue

            if rule.get("and_no_consequence") and consequences:
                continue

            if rule.get("to"):
                severity = str(rule["to"])
            elif rule.get("maximum"):
                severity = at_most(severity, str(rule["maximum"]))

        return severity if severity in level_index else "info"

    def determine_actual_incident(
        self,
        *,
        article_role: str,
        confidence: float,
        matched_rule_id: str | None,
        negative_terms: Sequence[str],
    ) -> tuple[bool, str | None]:
        classification = self.runtime.get("classification", {})
        threshold = float(
            classification.get("minimum_incident_confidence", 0.72)
        )

        non_incident_roles = {
            "reaction",
            "analysis",
            "summary",
            "procurement",
            "exercise",
            "historical",
        }

        if article_role in non_incident_roles:
            return False, f"article_role:{article_role}"
        if negative_terms:
            return False, "negative_context"
        if not matched_rule_id:
            return False, "no_composition_rule"
        if confidence < threshold:
            return False, f"confidence_below_{threshold:.2f}"
        return True, None

    # ------------------------------------------------------------------
    # Fingerprint and identity
    # ------------------------------------------------------------------

    def generate_fingerprint(
        self,
        *,
        country: str | None,
        family: str,
        subcategory: str,
        subtype: str,
        object_name: str,
        action: str,
        actor: str,
        target: str,
        place: str | None,
        published: str | None,
    ) -> dict[str, Any]:
        fallback_place = country if not place else place
        values = {
            "country": country or "unknown",
            "family": family,
            "subcategory": subcategory,
            "subtype": subtype,
            "object": object_name,
            "action": action,
            "actor": actor or "unknown",
            "target": target or "unknown",
            "place": fallback_place or "unknown",
            "event_date": _date_bucket(published),
        }

        fields = self.fingerprint_model.get(
            "fields",
            list(values.keys()),
        )
        fingerprint = {
            field: values.get(field, "unknown")
            for field in fields
        }

        canonical = "|".join(
            f"{key}={self.normalize_text(str(value))}"
            for key, value in fingerprint.items()
        )
        fingerprint["hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:24]
        return fingerprint

    def generate_incident_id(
        self,
        *,
        url: str,
        title: str,
        published: str | None,
        fingerprint: Mapping[str, Any],
    ) -> str:
        basis = url.strip() or "|".join(
            [
                title.strip(),
                published or "",
                str(fingerprint.get("hash", "")),
            ]
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_key(matches: Mapping[str, Sequence[str]]) -> str:
        return next(iter(matches), "unknown")

    @staticmethod
    def _flatten_matches(
        matches: Mapping[str, Sequence[str]],
    ) -> list[str]:
        flattened: list[str] = []
        for canonical, terms in matches.items():
            flattened.append(canonical)
            flattened.extend(terms)
        return _stable_unique(flattened)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_classifier: SecurityClassifier | None = None


def get_default_classifier() -> SecurityClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SecurityClassifier()
    return _default_classifier


def classify_security_incident(
    *,
    title: str,
    summary: str = "",
    source: str = "",
    url: str = "",
    published: str | None = None,
    country: str | None = None,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    source_is_official: bool = False,
    source_count: int = 1,
) -> SecurityIncident:
    """Simple importable function for fetch_local_sources.py."""
    return get_default_classifier().classify(
        title=title,
        summary=summary,
        source=source,
        url=url,
        published=published,
        country=country,
        city=city,
        latitude=latitude,
        longitude=longitude,
        source_is_official=source_is_official,
        source_count=source_count,
    )


# ---------------------------------------------------------------------------
# CLI and smoke test support
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a security-related article."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--published")
    parser.add_argument("--country")
    parser.add_argument("--city")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--official-source", action="store_true")
    parser.add_argument("--source-count", type=int, default=1)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    classifier = SecurityClassifier(args.taxonomy)
    incident = classifier.classify(
        title=args.title,
        summary=args.summary,
        source=args.source,
        url=args.url,
        published=args.published,
        country=args.country,
        city=args.city,
        latitude=args.lat,
        longitude=args.lon,
        source_is_official=args.official_source,
        source_count=max(1, args.source_count),
    )

    indent = None if args.compact else 2
    print(
        json.dumps(
            incident.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
