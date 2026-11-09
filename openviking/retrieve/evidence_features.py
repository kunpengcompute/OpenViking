# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared lexical features and source identity helpers for evidence retrieval."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from openviking.utils.spacy_utils import detect_language, load_spacy_model
from openviking_cli.utils.config import get_openviking_config

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Deterministic lexical / entity extraction (ported from index_proto.py)
# ---------------------------------------------------------------------------
_STOP = set(
    """i me my we our you your he his she her it its they their them a an the and or
    but if then so of in on at to for with from by as is are was were be been being
    have has had do does did will would can could should shall may might must about
    into over under again further once here there all any both each few more most
    other some such no nor not only own same than too very s t can now just don
    really lot lots yeah oh hey wow thanks thank great good going go get got want
    gonna im thats dont know tell let us """.split()
)
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "NORP", "WORK_OF_ART"}
_KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
# Event hyperedge helpers ------------------------------------------------------
# Deterministic event-card date normalization (probe phase, zero LLM).
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _normalize_date(text: str) -> Optional[str]:
    """Normalize a DATE entity to ``YYYY-MM-DD`` (or None when unparseable)."""
    s = (text or "").strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"([a-z]+) (\d{1,2}),? (\d{4})", s)  # may 8 2023 / may 8, 2023
    if m:
        mn = _MONTHS.get(m.group(1)) or _MONTH_ABBR.get(m.group(1))
        if mn:
            return f"{m.group(3)}-{mn:02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", s)  # 8 may 2023
    if m:
        mn = _MONTHS.get(m.group(2)) or _MONTH_ABBR.get(m.group(2))
        if mn:
            return f"{m.group(3)}-{mn:02d}-{int(m.group(1)):02d}"
    return None


@dataclass
class EventNode:
    """One event hyperedge derived from a deterministic event card."""

    id: str
    date: str  # YYYY-MM-DD (card directory date)
    name: str
    ents: Set[str] = field(default_factory=set)  # participants (lowercased)
    keys: Set[str] = field(default_factory=set)  # lexical keys of the summary
    member_turns: List[str] = field(default_factory=list)  # episode:turn ids

    @property
    def has_members(self) -> bool:
        return bool(self.member_turns)


def _get_nlp(text: str = ""):
    """Load the configured model for the text's language; default to English."""
    config = get_openviking_config().memory
    model = config.spacy_model_zh if detect_language(text) == "zh" else config.spacy_model_en
    return load_spacy_model(model)


def extract_keys(text: str) -> Tuple[List[str], Set[str]]:
    """Return (entities, lexical keys) for one turn text.
    Deterministic, zero LLM. Mirrors index_proto.extract_keys: spaCy NER
    entities (1-4 tokens, whitelisted labels) + lemma unigrams/bigrams.
    """
    nlp = _get_nlp(text)
    if nlp is None:
        return [], set()
    try:
        doc = nlp(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[evidence] spaCy parse failed: %s", exc)
        return [], set()
    ents = []
    for e in doc.ents:
        if e.label_ in _ENTITY_LABELS:
            t = e.text.lower().strip()
            if 1 <= len(t.split()) <= 4:
                ents.append(t)
    toks = [t for t in doc if t.is_alpha and t.text.lower() not in _STOP and len(t.text) > 1]
    min_length = 2 if detect_language(text) == "zh" else 3
    lemmas = [
        (t.lemma_ or t.text).lower()
        for t in toks
        if t.pos_ in _KEEP_POS and len(t.lemma_ or t.text) >= min_length
    ]
    bigrams = [f"{lemmas[i]}_{lemmas[i + 1]}" for i in range(len(lemmas) - 1)]
    keys = set(lemmas) | set(bigrams)
    return ents, keys


def extract_labeled_ents(text: str) -> Dict[str, Set[str]]:
    """One spaCy pass returning whitelisted entities grouped by label.
    Used by the event hyperedge channel to distinguish DATE from PERSON
    entities at query time (``extract_keys`` deliberately drops labels).
    Deterministic, zero LLM.
    """
    nlp = _get_nlp(text)
    if nlp is None:
        return {}
    try:
        doc = nlp(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[evidence] labeled entity parse failed: %s", exc)
        return {}
    out: Dict[str, Set[str]] = defaultdict(set)
    for e in doc.ents:
        t = e.text.strip().lower()
        if t and e.label_ in _ENTITY_LABELS:
            out[e.label_].add(t)
    return dict(out)


_EPISODE_RE = re.compile(r"^sample(?P<sid>[0-9_]+)_s(?P<sess>\d+)_c(?P<chunk>\d+)$")


def parse_episode_key(session_id: str) -> Optional[Tuple[str, str, int]]:
    """Parse a chunk session_id into (episode_key, session_label, chunk_idx).
    Accepts ``sample{sid}_s{sess}_c{chunk}`` encoded by import_to_ov.py.
    Returns None for ordinary session IDs, which append using stable message identities.
    """
    m = _EPISODE_RE.match(session_id or "")
    if not m:
        return None
    sid, sess, chunk = m.group("sid"), m.group("sess"), int(m.group("chunk"))
    return (f"sample{sid}_s{sess}", f"s{sess}", chunk)


# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------
