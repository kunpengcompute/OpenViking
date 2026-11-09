# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source-backed facts, entities and completed events without candidate LLM calls."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking.session.memory.atomic_facts import select_assistant_facts, select_user_facts
from openviking.session.memory.dataclass import ResolvedOperation, ResolvedOperations
from openviking.session.memory.deterministic_rules import (
    _CODE_IP_RE,
    _CODE_PATH_RE,
    _CODE_SYMBOL_RE,
    _CODE_URL_RE,
    _CODE_VERSION_RE,
    _ENTITY_LABELS,
    _EVENT_GOALS,
    _EVENT_VERB_PATTERNS,
    _LABEL_CATEGORY,
    _MONEY_RE,
    _PERCENT_RE,
    _QUANTITY_RE,
    _TITLE_RE,
    _build_entity_content,
    _build_event_summary,
    _build_ranges,
    _canonicalize,
    _confidence,
    _EntityHit,
    _EntityRecord,
    _event_name_for,
    _EventRecord,
    _ExtractedMsg,
    _merge_entity_content,
    _merge_title_entities,
    _resolve_dates_in_text,
    _split_speaker,
)
from openviking.session.memory.fact_records import (
    _parse,
    alias_in_text,
    make_fact_records,
    merge_fact_records,
)
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.memory.utils.uri import generate_uri
from openviking.utils.spacy_utils import detect_language, load_spacy_model
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.exceptions import NotFoundError
from openviking_cli.utils import get_logger

logger = get_logger(__name__)
ENTITIES_MEMORY_TYPE = "entities"
EVENTS_MEMORY_TYPE = "events"
FACTS_MEMORY_TYPE = "facts"


class DeterministicMemoryExtractor:
    """Zero-LLM memory extractor producing ``ResolvedOperations``."""

    def __init__(
        self,
        registry: Any,
        viking_fs: Any,
        ner_model: str = "zh_core_web_sm",
        ner_model_en: str = "en_core_web_sm",
        *,
        isolation_handler: Any = None,
        max_entities: int = 30,
        max_events_per_batch: int = 20,
        max_events_per_message: int = 3,
    ):
        self._isolation_handler = isolation_handler
        self._registry = registry
        self._viking_fs = viking_fs
        self._ner_models = {"zh": ner_model, "en": ner_model_en}
        self._max_entities = max_entities
        self._max_events_per_batch = max_events_per_batch
        self._max_events_per_message = max_events_per_message

    # -- public -------------------------------------------------------------
    async def extract(
        self,
        messages: List[Any],
        ctx: Any,
        allowed_memory_types: Optional[Set[str]],
        extract_context: Any,
    ) -> ResolvedOperations:
        """Extract entity + event memories deterministically.
        ``messages`` here must be ``extract_context.messages`` so that event
        ``ranges`` indices match what the events content template renders at
        apply time.
        """
        extracted = self._prepare_messages(messages, extract_context)
        if not extracted:
            return ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
        if ctx is None or not getattr(ctx, "user", None):
            logger.warning("Deterministic extraction skipped: missing ctx.user")
            return ResolvedOperations(upsert_operations=[], delete_file_contents=[], errors=[])
        if self._isolation_handler is None:
            return await self._extract_prepared(
                extracted, ctx, allowed_memory_types, extract_context
            )

        from copy import copy

        from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler

        # Partition before any reads or fact attachment: one peer's statements
        # must not enter another peer's card even when entity names coincide.
        groups: Dict[str, List[_ExtractedMsg]] = {}
        target = None
        for msg in extracted:
            if msg.role == "user":
                target = self._isolation_handler._message_target_id(messages[msg.index])
            if target is not None:
                groups.setdefault(target, []).append(msg)
        upserts, errors = [], []
        for target, group in groups.items():
            extractor = copy(self)
            extractor._isolation_handler = MemoryIsolationHandler(
                ctx,
                extract_context,
                allowed_memory_types=allowed_memory_types,
                allow_self=target == "__self",
                allowed_peer_ids={target} if target != "__self" else set(),
                peer_memory_enabled=self._isolation_handler.peer_memory_enabled,
            )
            result = await extractor._extract_prepared(
                group, ctx, allowed_memory_types, extract_context
            )
            upserts.extend(result.upsert_operations)
            errors.extend(result.errors)
        return ResolvedOperations(upsert_operations=upserts, delete_file_contents=[], errors=errors)

    async def _extract_prepared(self, extracted, ctx, allowed_memory_types, extract_context):
        user_id = ctx.user.user_id
        errors: List[str] = []
        upserts: List[ResolvedOperation] = []
        if self._type_allowed(allowed_memory_types, FACTS_MEMORY_TYPE):
            try:
                upserts.extend(self._extract_atomic_facts(extracted, user_id, extract_context))
            except Exception as exc:
                message = f"Deterministic fact extraction failed: {exc}"
                logger.exception(message)
                errors.append(message)
        if self._type_allowed(allowed_memory_types, ENTITIES_MEMORY_TYPE):
            try:
                upserts.extend(
                    await self._extract_entities(extracted, ctx, user_id, extract_context)
                )
            except Exception as exc:  # pragma: no cover - defensive
                message = f"Deterministic entity extraction failed: {exc}"
                logger.exception(message)
                errors.append(message)
        if self._type_allowed(allowed_memory_types, EVENTS_MEMORY_TYPE):
            try:
                upserts.extend(self._extract_events(extracted, ctx, user_id, extract_context))
            except Exception as exc:  # pragma: no cover - defensive
                message = f"Deterministic event extraction failed: {exc}"
                logger.exception(message)
                errors.append(message)
        if self._isolation_handler is not None:
            for operation in upserts:
                operation.uris = self._isolation_handler.calculate_memory_uris(
                    self._registry.get(operation.memory_type), operation, extract_context
                )
        for operation in upserts:
            operation.memory_fields["extraction_method"] = "deterministic"
        return ResolvedOperations(
            upsert_operations=upserts,
            delete_file_contents=[],
            errors=errors,
            resolved_links=[],
        )

    # -- N1: input ----------------------------------------------------------
    def _prepare_messages(
        self, messages: List[Any], extract_context: Any = None
    ) -> List[_ExtractedMsg]:
        result: List[_ExtractedMsg] = []
        grouped: Dict[str, _ExtractedMsg] = {}
        for index, message in enumerate(messages or []):
            role = str(getattr(message, "role", "") or "").lower()
            if role not in ("user", "assistant"):
                continue
            text = self._message_text(message)
            if not text:
                continue
            ts = _parse_created_at(getattr(message, "created_at", None))
            meta = (getattr(extract_context, "chunk_meta", None) or {}).get(id(message))
            source_id = getattr(meta, "source_message_id", None) or getattr(message, "id", "")
            if meta is not None and source_id in grouped:
                grouped[source_id].text += text
                grouped[source_id].indices.append(index)
                continue
            speaker, _ = _split_speaker(text)
            speaker = speaker or getattr(message, "peer_id", None) or role
            item = _ExtractedMsg(
                index=index,
                role=role,
                ts=ts,
                text=text,
                source_id=source_id,
                speaker=speaker,
                indices=[index],
            )
            result.append(item)
            if meta is not None:
                grouped[source_id] = item
        return result

    @staticmethod
    def _message_text(message: Any) -> str:
        texts = [
            str(part.text)
            for part in getattr(message, "parts", []) or []
            if getattr(part, "text", None)
        ]
        return "\n".join(texts) if texts else str(getattr(message, "content", "") or "")

    # -- helpers ------------------------------------------------------------
    def _type_allowed(self, allowed_memory_types: Optional[Set[str]], memory_type: str) -> bool:
        if allowed_memory_types is not None and memory_type not in allowed_memory_types:
            return False
        schema = self._registry.get(memory_type)
        return bool(schema is not None and getattr(schema, "enabled", True))

    def _extract_atomic_facts(
        self, extracted: List[_ExtractedMsg], user_id: str, extract_context: Any
    ) -> List[ResolvedOperation]:
        schema = self._registry.get(FACTS_MEMORY_TYPE)
        operations: List[ResolvedOperation] = []
        previous_user = ""
        seen: Set[str] = set()
        for msg in extracted:
            if msg.role == "user":
                rows = select_user_facts(msg.text, msg.source_id, msg.ts, msg.speaker)
                previous_user = msg.text
                # A personal update is more useful than conversational filler.
                rows.sort(
                    key=lambda row: (
                        row["quantity"] is not None,
                        row["time_expression"] is not None,
                        row["fact"]["predicate"] != "states",
                    ),
                    reverse=True,
                )
                rows = rows[:4]
            else:
                rows = select_assistant_facts(msg.text, previous_user, msg.source_id, msg.ts)
                rows = rows[:6]
            for row in rows:
                fact = row["fact"]
                if fact["id"] in seen:
                    continue
                seen.add(fact["id"])
                role = "assistant" if msg.role == "assistant" else "user"
                source = fact["source_turn_ids"][0]
                content = f"{role.title()} statement: {row['content'][:300]}\nSource: {source}"
                if fact.get("recorded_at"):
                    content += f"\nRecorded: {fact['recorded_at']}"
                if row.get("context"):
                    content += f"\nEarlier in the same source message: {row['context'][:100]}"
                if row.get("quantity"):
                    quantity = row["quantity"]
                    object_text = f" {quantity['object']}" if quantity["object"] else ""
                    content += (
                        f"\nQuantity: {quantity['value']}{object_text} ({quantity['status']})"
                    )
                if row.get("scope"):
                    content += f"\nScope: {row['scope']}"
                if row.get("time_expression"):
                    content += f"\nTime expression: {row['time_expression']}"
                if fact.get("time"):
                    content += f"\nEvent time: {fact['time']}"
                content = content[:600]
                memory_fields = {
                    "ranges": _build_ranges(msg.indices),
                    "speaker": role,
                    "fact_id": fact["id"],
                    "content": content,
                    "fact_records": [fact],
                    "source_turn_ids": fact["source_turn_ids"],
                    "recorded_at": fact["recorded_at"],
                    "assertion_type": fact["assertion_type"],
                    "quantity": row.get("quantity"),
                    "scope": row.get("scope"),
                    "time_expression": row.get("time_expression"),
                    "user_id": user_id,
                }
                uri = generate_uri(
                    schema,
                    {"speaker": role, "fact_id": fact["id"]},
                    user_space=user_id,
                    extract_context=extract_context,
                )
                operations.append(
                    ResolvedOperation(
                        old_memory_file_content=None,
                        memory_fields=memory_fields,
                        memory_type=FACTS_MEMORY_TYPE,
                        uris=[uri],
                    )
                )
        return operations

    def _get_nlp(self, lang: str = "zh"):
        return load_spacy_model(self._ner_models[lang])

    async def _entity_exists(self, uri: str, ctx: Any) -> bool:
        return bool(await self._viking_fs.exists(uri, ctx=ctx))

    async def _read_entity_file(self, uri: str, ctx: Any) -> Optional[Any]:
        try:
            raw = await self._viking_fs.read_file(uri, ctx=ctx)
        except (NotFoundError, FileNotFoundError):
            return None
        if not raw:
            return None
        return MemoryFileUtils.read(raw, uri=uri)

    # -- entities -----------------------------------------------------------
    async def _extract_entities(
        self,
        extracted: List[_ExtractedMsg],
        ctx: Any,
        user_id: str,
        extract_context: Any,
    ) -> List[ResolvedOperation]:
        schema = self._registry.get(ENTITIES_MEMORY_TYPE)
        if schema is None:
            return []
        hits = self._collect_entity_hits(extracted)
        records: Dict[str, _EntityRecord] = {}
        for hit in hits:
            canonical = _canonicalize(hit.text)
            if not canonical or canonical in ("用户", "我", "我们", "你"):
                continue
            record = records.setdefault(
                canonical,
                _EntityRecord(canonical=canonical, category=hit.category),
            )
            record.signal_types.update(hit.signals)
            record.msg_indices.add(hit.msg_index)
            record.aliases.add(hit.text)
            if hit.category == "organization":
                record.orgs.add(hit.text)
        # Collect per-record fact sentences + relation pairs.
        self._attach_entity_facts(records, extracted)
        self._attach_entity_relations(records, extracted)
        # Cross-window confirmation: in-batch multi-signal / multi-message votes.
        confirmed = {
            key: record for key, record in records.items() if self._in_batch_confirmed(record)
        }
        # Disk check for everything else: an existing card is a prior window.
        schema_uris: Dict[str, str] = {}
        for key in list(records):
            if key in confirmed:
                continue
            uri = self._entity_uri(schema, records[key], user_id, extract_context)
            schema_uris[key] = uri
            if uri and await self._entity_exists(uri, ctx):
                confirmed[key] = records[key]
        if not confirmed:
            return []
        ops: List[ResolvedOperation] = []
        ordered = sorted(confirmed.items(), key=lambda item: -len(item[1].msg_indices))[
            : self._max_entities
        ]
        for key, record in ordered:
            uri = schema_uris.get(key) or self._entity_uri(schema, record, user_id, extract_context)
            old_file = await self._read_entity_file(uri, ctx) if uri else None
            aliases = set(record.aliases)
            if old_file is not None:
                old_aliases = (old_file.extra_fields or {}).get("aliases") or []
                if isinstance(old_aliases, list):
                    aliases.update(str(item) for item in old_aliases)
            content = _build_entity_content(record, record.facts)
            if old_file is not None:
                old_content = old_file.plain_content()
                old_facts = (old_file.extra_fields or {}).get("fact_records", [])
                if not old_facts or any(
                    f.get("version") != 2 for f in old_facts if isinstance(f, dict)
                ):
                    # Legacy relations have not passed the current phrase parser.
                    old_content = "\n".join(
                        line
                        for line in old_content.splitlines()
                        if not re.match(r"^- 与.+：", line)
                    )
                content = _merge_entity_content(old_content, content)
            fact_records = merge_fact_records(
                (old_file.extra_fields or {}).get("fact_records", []) if old_file else [],
                record.fact_records,
            )
            memory_fields: Dict[str, Any] = {
                "ranges": _build_ranges(sorted(record.msg_indices)),
                "category": record.category,
                "name": key,
                "content": content,
                "aliases": sorted(aliases),
                "confidence": _confidence(key),
                "user_id": user_id,
                "fact_records": fact_records,
                "source_turn_ids": sorted(
                    set(
                        (old_file.extra_fields or {}).get("source_turn_ids", []) if old_file else []
                    )
                    | {
                        msg.source_id
                        for msg in extracted
                        if msg.source_id
                        and set(msg.indices or [msg.index]) & set(record.msg_indices)
                    }
                ),
            }
            ops.append(
                ResolvedOperation(
                    old_memory_file_content=None,
                    memory_fields=memory_fields,
                    memory_type=ENTITIES_MEMORY_TYPE,
                    uris=[uri],
                )
            )
        return ops

    def _entity_uri(
        self, schema: Any, record: _EntityRecord, user_space: str, extract_context: Any
    ) -> str:
        fields = {
            "category": record.category,
            "name": record.canonical,
            "ranges": _build_ranges(sorted(record.msg_indices)),
        }
        if self._isolation_handler is not None:
            operation = ResolvedOperation(
                old_memory_file_content=None,
                memory_fields=fields,
                memory_type=ENTITIES_MEMORY_TYPE,
                uris=[],
            )
            uris = self._isolation_handler.calculate_memory_uris(schema, operation, extract_context)
            # Multiple ownership targets must not share a prefetched snapshot.
            return uris[0] if len(uris) == 1 else ""
        return generate_uri(schema, fields, user_space=user_space, extract_context=extract_context)

    def _collect_entity_hits(self, extracted: List[_ExtractedMsg]) -> List[_EntityHit]:
        hits: List[_EntityHit] = []
        for msg in extracted:
            # An explicit speaker is an identified entity even with one message.
            if msg.speaker and msg.speaker not in ("user", "assistant"):
                hits.append(
                    _EntityHit(
                        text=msg.speaker,
                        category="person",
                        msg_index=msg.index,
                        signals={"speaker", "source"},
                    )
                )
            nlp = self._get_nlp(detect_language(msg.text))
            if nlp is not None:
                doc = nlp(msg.text)
                ner_entities = [ent.text for ent in doc.ents if ent.label_ in _ENTITY_LABELS]
                ner_entities = _merge_title_entities(ner_entities, msg.text)
                for entity in ner_entities:
                    label = _LABEL_CATEGORY.get(
                        next(
                            (ent.label_ for ent in doc.ents if ent.text == entity),
                            "PERSON",
                        ),
                        "person",
                    )
                    hits.append(
                        _EntityHit(
                            text=entity,
                            category=label,
                            msg_index=msg.index,
                            signals={"ner"},
                        )
                    )
            # Relation-lexicon signal (N7): 称呼 like 张总/王老师 are a
            # person signal independent of NER, so a single message mention
            # can reach the multi-signal confirmation threshold.
            hits.extend(
                _EntityHit(
                    text=match.group(0),
                    category="person",
                    msg_index=msg.index,
                    signals={"relation"},
                )
                for match in _TITLE_RE.finditer(msg.text)
                if match.group(0).strip()
            )
            # Code-domain + quantity regex fallback (NER-blind to these).
            hits.extend(
                _EntityHit(
                    text=match.group(0),
                    category="code",
                    msg_index=msg.index,
                    signals={"regex"},
                )
                for pattern in (
                    _CODE_URL_RE,
                    _CODE_PATH_RE,
                    _CODE_VERSION_RE,
                    _CODE_IP_RE,
                    _CODE_SYMBOL_RE,
                )
                for match in pattern.finditer(msg.text)
                if match.group(0).strip()
            )
            hits.extend(
                _EntityHit(
                    text=match.group(0),
                    category="quantity",
                    msg_index=msg.index,
                    signals={"regex"},
                )
                for pattern in (_QUANTITY_RE, _PERCENT_RE, _MONEY_RE)
                for match in pattern.finditer(msg.text)
            )
        # Deduplicate same text+category+message.
        seen: Set[Tuple[str, str, int]] = set()
        deduped: List[_EntityHit] = []
        for hit in hits:
            key = (hit.text, hit.category, hit.msg_index)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    def _attach_entity_facts(
        self, records: Dict[str, _EntityRecord], extracted: List[_ExtractedMsg]
    ) -> None:
        for msg in extracted:
            facts = make_fact_records(
                msg.text, msg.speaker, msg.source_id, msg.ts, _resolve_dates_in_text
            )
            for fact in facts:
                quote = fact["source_quotes"][msg.source_id]
                for record in records.values():
                    aliases = record.aliases | {record.canonical}
                    if any(
                        alias_in_text(a, quote, record.category)
                        or alias_in_text(a, fact["subject"], record.category)
                        or a.replace("_", " ").casefold() == msg.speaker.casefold()
                        for a in aliases
                    ):
                        if quote not in record.facts:
                            record.facts.append(quote)
                        record.fact_records = merge_fact_records(record.fact_records, [fact])

    def _attach_entity_relations(
        self, records: Dict[str, _EntityRecord], extracted: List[_ExtractedMsg]
    ) -> None:
        for record in records.values():
            for fact in record.fact_records:
                if (
                    not (fact["predicate"].startswith("has_") or fact["predicate"].endswith("_of"))
                    or fact["assertion_type"] != "fact"
                ):
                    continue
                if _canonicalize(fact["subject"]) != record.canonical:
                    continue
                pair = (_canonicalize(fact["object"]), fact["predicate"])
                if pair not in record.relations:
                    record.relations.append(pair)

    @staticmethod
    def _in_batch_confirmed(record: _EntityRecord) -> bool:
        if len(record.signal_types) >= 2:
            return True
        if len(record.msg_indices) >= 2:
            return True
        return False

    # -- events -------------------------------------------------------------
    def _extract_events(
        self,
        extracted: List[_ExtractedMsg],
        ctx: Any,
        user_id: str,
        extract_context: Any,
    ) -> List[ResolvedOperation]:
        schema = self._registry.get(EVENTS_MEMORY_TYPE)
        if schema is None:
            return []
        events: List[_EventRecord] = []
        for msg in extracted:
            per_message: List[_EventRecord] = []
            facts = make_fact_records(
                msg.text, msg.speaker, msg.source_id, msg.ts, _resolve_dates_in_text
            )
            for fact in facts:
                for event_type in _EVENT_VERB_PATTERNS:
                    if not is_completed_event(fact, event_type):
                        continue
                    per_message.append(
                        _EventRecord(
                            event_type=event_type,
                            fact=fact["source_quotes"][msg.source_id],
                            participants=[fact["subject"]],
                            dates=[fact["time"]] if fact["time"] else [],
                            msg_indices=msg.indices or [msg.index],
                            fact_records=[fact],
                        )
                    )
                    break
            events.extend(per_message[: self._max_events_per_message])
            if len(events) >= self._max_events_per_batch:
                break
        ops: List[ResolvedOperation] = []
        for event in events[: self._max_events_per_batch]:
            ranges = self._event_ranges(event, extracted)
            if not ranges:
                continue
            event_name = (
                _event_name_for(event.event_type, event.fact, event.participants)
                + "_"
                + event.fact_records[0]["id"][:12]
            )
            if not event_name:
                continue
            memory_fields: Dict[str, Any] = {
                "event_name": event_name,
                "event_type": event.event_type,
                "fact_records": event.fact_records,
                "goal": _EVENT_GOALS[event.event_type],
                "summary": _build_event_summary(event, event.fact),
                "ranges": ranges,
                "user_id": user_id,
            }
            try:
                uri = generate_uri(
                    schema,
                    {"event_name": event_name, "ranges": ranges},
                    user_space=user_id,
                    extract_context=extract_context,
                )
            except Exception:
                logger.warning(
                    "Skipping deterministic event: uri generation failed for %r",
                    event_name,
                )
                continue
            ops.append(
                ResolvedOperation(
                    old_memory_file_content=None,
                    memory_fields=memory_fields,
                    memory_type=EVENTS_MEMORY_TYPE,
                    uris=[uri],
                )
            )
        return ops

    def _event_ranges(self, event: _EventRecord, extracted: List[_ExtractedMsg]) -> str:
        """Build ranges that always include a user message (events.yaml rule)."""
        msg_index = event.msg_indices[0]
        msg = next((m for m in extracted if m.index == msg_index), None)
        if msg is None:
            return ""
        if msg.role == "user":
            return _build_ranges(event.msg_indices)
        # Assistant-only hit: include the nearest preceding user message.
        for candidate in reversed(extracted):
            if candidate.index < msg_index and candidate.role == "user":
                return _build_ranges([candidate.index, *event.msg_indices])
        return ""


def _parse_created_at(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return parse_iso_datetime(str(raw))
    except (ValueError, TypeError):
        return None


def is_completed_event(fact: Dict[str, Any], event_type: str) -> bool:
    """Require a source assertion of an occurrence, not a wish, habit or question."""
    if not isinstance(event_type, str) or event_type not in _EVENT_VERB_PATTERNS:
        return False
    if fact.get("assertion_type") != "fact" or fact.get("predicate") in {
        "states",
        "want",
        "hope",
        "plan",
        "love",
        "like",
        "enjoy",
        "remember",
        "say",
        "tell",
        "think",
        "hear",
    }:
        return False
    if fact.get("predicate", "").startswith("has_") or fact.get("predicate", "").endswith("_of"):
        return False
    quote = next(iter(fact.get("source_quotes", {}).values()), "")
    if not any(pattern.search(quote) for pattern in _EVENT_VERB_PATTERNS.get(event_type, [])):
        return False
    if re.search(r"[\u4e00-\u9fff]", quote):
        return fact["predicate"] in {
            "去",
            "到",
            "飞",
            "搬到",
            "搬去",
            "参加",
            "见",
            "庆祝",
            "住院",
            "做",
        } and bool(re.search(r"了|过", quote))
    actions = {
        "travel": {
            "go",
            "travel",
            "fly",
            "camp",
            "hike",
            "visit",
            "take",
            "explore",
            "bike",
            "drive",
        },
        "meeting": {"meet", "gather", "attend", "join"},
        "move": {"move", "relocate"},
        "job_change": {
            "quit",
            "resign",
            "promote",
            "hire",
            "start",
            "switch",
            "join",
            "interview",
            "pass",
        },
        "birthday_celebration": {"celebrate", "have", "host", "throw", "attend"},
        "commitment": {"promise", "commit", "decide", "confirm", "agree"},
        "health": {"get", "fall", "undergo", "have", "go"},
    }
    if fact["predicate"] not in actions.get(event_type, set()):
        return False
    if event_type == "job_change" and not re.search(
        r"\b(job|work|employment|employer|company|career|salary|position)\b", quote, re.I
    ):
        return False
    doc = _parse(quote)
    if doc is None:
        return False
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    return root is not None and (
        root.tag_ == "VBD"
        or (root.tag_ == "VBN" and any(c.lemma_ == "have" for c in root.children))
    )
