# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lexical, semantic and episode scoring for the source-backed evidence index."""

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openviking.retrieve.evidence_features import (
    _normalize_date,
    extract_keys,
    extract_labeled_ents,
)

logger = logging.getLogger(__name__)


class EvidenceRanking:
    DEFAULT_EPI_WINDOW = 4
    DEFAULT_TOP_K = 15
    DEFAULT_SEM_ANCHOR_TOP = 3
    DEFAULT_CLOSURE_RADIUS = 1

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        epi_window: int = DEFAULT_EPI_WINDOW,
        propagate: bool = True,
        use_semantic: bool = False,
        q_vec: Optional[List[float]] = None,
        sem_anchor_top: int = DEFAULT_SEM_ANCHOR_TOP,
        closure_radius: int = DEFAULT_CLOSURE_RADIUS,
        use_events: bool = False,
        event_top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Rank raw turns for ``query`` and return an evidence package.
        Ported from index_proto.retrieve: lexical (df-filtered) + entity
        signals, dedup-session top-3 primary evidence, episode window
        expansion (±epi_window neighbors), co-occurrence propagation, then a
        deterministic merge. Returned items keep raw text so the caller can
        build a prompt without a second read.
        With ``use_semantic=True`` and a ``q_vec`` the ALG-A path runs instead:
        semantic cosine anchors (top-``sem_anchor_top`` turns with an ``emb``)
        join the window/co-occurrence anchor sets, and the anchors plus their
        ±``closure_radius`` neighbors are force-included in the package
        (Zero-Mem Eq.14 / EdgeMem inclusion closure). If no turn has an
        ``emb`` or ``q_vec`` is missing, this is the version-1 path unchanged.
        With ``use_events=True`` and event nodes attached (see
        ``load_events_from_memories``), event hyperedges contribute a fourth
        anchor source: events scored by date / participant / lexical-key
        overlap, member turns boosted, and date-matched event members
        force-included (EdgeMem Time-hyperedge style). Off by default: when no
        event matches, the output is byte-identical to the event-less path.
        """
        if not self._n_turns or not query.strip():
            return []
        n = self._n_turns
        q_ents, q_keys = extract_keys(query)

        def idf(df):
            return math.log(n / (df + 1)) + 1.0

        # Low-discrimination key filter (mirrors index_proto):
        # unigram df>25% dropped; entities df>40% dropped (protagonist names).
        q_keys = {k for k in q_keys if "_" in k or len(self._lex_inv.get(k, ())) <= n * 0.25}
        q_ents = {e for e in q_ents if len(self._ent_inv.get(e, ())) <= n * 0.40}
        score: Dict[str, float] = defaultdict(float)
        for k in q_keys:
            df = len(self._lex_inv.get(k, ()))
            if df == 0:
                continue
            w = idf(df) * (1.5 if "_" in k else 1.0)
            for tid in self._lex_inv[k]:
                score[tid] += w
        for e in q_ents:
            df = len(self._ent_inv.get(e, ()))
            if df == 0:
                continue
            w = idf(df) * 1.2
            for tid in self._ent_inv[e]:
                score[tid] += w
        if not score:
            return []
        # Primary evidence: top-3 across distinct episodes (deterministic).
        primary = []
        seen_ep = set()
        for tid in sorted(score, key=lambda t: (-score.get(t, 0.0), t)):
            ep = tid.rsplit(":", 1)[0]
            if ep in seen_ep:
                continue
            seen_ep.add(ep)
            primary.append(tid)
            if len(primary) >= 3:
                break
        # Semantic anchors (ALG-A): top-K turns by cosine against the query
        # vector. Empty when the semantic channel is off / unavailable.
        sem_anchors: List[str] = []
        if use_semantic and q_vec is not None:
            sem_anchors = self._sem_anchors(q_vec, sem_anchor_top)
        sem_active = bool(sem_anchors)
        anchors = list(dict.fromkeys(primary + sem_anchors))
        lex_ch = sorted(score, key=lambda t: (-score.get(t, 0.0), t))[:top_k]
        lex_set = set(lex_ch)
        # Episode window expansion: neighbors of the anchor turns. The ALG-A
        # path applies the bonus directly (validated on the pilot harness);
        # the legacy path keeps the v1 lex_set skip for byte-identical output.
        if sem_active:
            for anchor in anchors:
                ep, idx_s = anchor.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - epi_window), idx + epi_window + 1):
                    oid = self._turn_id(ep, j)
                    if oid not in self._turn_by_id:
                        continue
                    d = abs(j - idx)
                    bonus = 1.0 - d / (epi_window + 1)
                    score[oid] = max(score.get(oid, 0.0), bonus)
        else:
            epi_cand: Dict[str, float] = {}
            for tid in anchors:
                ep, idx_s = tid.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - epi_window), idx + epi_window + 1):
                    oid = self._turn_id(ep, j)
                    if oid not in self._turn_by_id:
                        continue
                    if oid in lex_set:
                        continue
                    d = abs(j - idx)
                    bonus = 1.0 - d / (epi_window + 1)
                    epi_cand[oid] = max(epi_cand.get(oid, 0.0), bonus)
            for oid, bonus in sorted(epi_cand.items(), key=lambda kv: -kv[1]):
                score[oid] = max(score.get(oid, 0.0), bonus)
        # Co-occurrence propagation: turns sharing a discriminative key with
        # an anchor turn.
        if propagate:
            for tid in anchors:
                disc = {
                    k
                    for k in self._turn_keys.get(tid, ())
                    if "_" in k or len(self._lex_inv.get(k, ())) <= n * 0.25
                }
                if not disc:
                    continue
                for ep, epd in self._episodes.items():
                    for turn in epd.get("turns", []):
                        oid = self._turn_id(ep, turn["turn"])
                        if oid in score:
                            continue
                        shared = set(turn.get("keys", ())) & disc
                        if shared:
                            score[oid] = max(score.get(oid, 0.0), 0.4 * len(shared))
        # Event hyperedge channel (zero LLM, off by default): events scored by
        # date / participant / lexical-key overlap; top-K event members get a
        # boost, date-matched event members are force-included below.
        event_anchors: List[str] = []
        if use_events and self._events:
            labeled = extract_labeled_ents(query)
            q_dates = set()
            for dt in labeled.get("DATE", ()):
                norm = _normalize_date(dt)
                if norm:
                    q_dates.add(norm)
            q_persons = labeled.get("PERSON", set())
            ev_scores: Dict[str, float] = {}
            for ev in self._events:
                s = 0.0
                date_hit = ev.date in q_dates
                if date_hit:
                    s += 3.0
                shared = q_keys & ev.keys
                # Protagonist names match dozens of events; a pure participant
                # hit is not enough on its own (requires a key overlap too).
                if ev.ents & q_persons and shared:
                    s += 1.5
                if shared:
                    s += sum(idf(len(self._lex_inv.get(k, ()))) for k in shared) * 1.2
                if s > 0:
                    ev_scores[ev.id] = s
            for ev_id in sorted(ev_scores, key=lambda e: -ev_scores[e])[:event_top_k]:
                ev = self._event_by_id[ev_id]
                s_e = ev_scores[ev_id]
                boost = 0.5 * s_e
                cap = os.environ.get("OV_EVENT_CAP")
                if cap:
                    boost = min(boost, float(cap))
                date_hit = ev.date in q_dates
                for tid in ev.member_turns:
                    if tid not in self._turn_by_id:
                        continue
                    if date_hit:
                        # Time-hyperedge: date-matched members join the package
                        # regardless of prior signal (force-included below).
                        score[tid] = max(score.get(tid, 0.0), boost)
                        event_anchors.append(tid)
                    elif tid in score:
                        # Non-date matches only boost turns that already carry
                        # a signal; never inject cold member turns.
                        score[tid] = max(score[tid], boost)
        ranked_ids = sorted(score, key=lambda t: (-score.get(t, 0.0), t))[:top_k]
        # ALG-A closure: force-include semantic anchors + their ±radius
        # neighbors, displacing the lowest-ranked non-closure members.
        if sem_active and closure_radius >= 0:
            closure = set()
            for anchor in sem_anchors:
                ep, idx_s = anchor.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - closure_radius), idx + closure_radius + 1):
                    oid = self._turn_id(ep, j)
                    if oid in self._turn_by_id:
                        closure.add(oid)
            for tid in sorted(closure):
                if tid not in ranked_ids:
                    for cand in reversed(ranked_ids):
                        if cand not in closure:
                            ranked_ids.remove(cand)
                            break
                    ranked_ids.append(tid)
            ranked_ids = sorted(ranked_ids, key=lambda t: (-score.get(t, 0.0), t))
        # Event hyperedge closure: date-matched event members are
        # force-included (inclusion, not scoring; mirrors the semantic closure).
        if event_anchors:
            eclosure = set(event_anchors)
            for tid in sorted(eclosure):
                if tid not in ranked_ids:
                    for cand in reversed(ranked_ids):
                        if cand not in eclosure:
                            ranked_ids.remove(cand)
                            break
                    ranked_ids.append(tid)
            ranked_ids = sorted(ranked_ids, key=lambda t: (-score.get(t, 0.0), t))
        return [self._turn_by_id[tid] for tid in ranked_ids[:top_k]]

    def get_turns(self, turn_ids: List[str]) -> List[Dict[str, Any]]:
        """Read evidence from this index only, without exposing stored vectors."""
        return [
            {k: v for k, v in self._turn_by_id[tid].items() if k not in {"emb", "embedding_state"}}
            for tid in turn_ids
            if tid in self._turn_by_id
        ]

    def retrieve_candidates(
        self, query: str, *, limit: int, q_vec: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Fuse independent lexical and semantic ranks before context expansion.

        Used by the opt-in coverage strategy. Unlike legacy retrieval, a
        lexical miss must not suppress an otherwise valid semantic channel.
        """
        if not query.strip() or not self._n_turns:
            return []
        limit = max(1, min(limit, 50))
        ents, keys = extract_keys(query)
        scores: Dict[str, float] = defaultdict(float)
        for key in keys:
            ids = self._lex_inv.get(key, ())
            if not ids or ("_" not in key and len(ids) > max(1, self._n_turns * 0.25)):
                continue
            weight = math.log(self._n_turns / (len(ids) + 1)) + 1
            for tid in ids:
                scores[tid] += weight * (1.5 if "_" in key else 1)
        lexical_ids = set(scores)
        for ent in ents:
            ids = self._ent_inv.get(ent, ())
            if not ids or len(ids) > max(1, self._n_turns * 0.4):
                continue
            for tid in ids:
                # When a relation/topic matched, entity presence may refine
                # those hits but must not turn unrelated mentions into a
                # second ranked channel that overwhelms specific evidence.
                if lexical_ids and tid not in lexical_ids:
                    continue
                scores[tid] += math.log(self._n_turns / (len(ids) + 1)) + 1
        lexical = sorted(scores, key=lambda tid: (-scores[tid], tid))[:limit]
        semantic = self._sem_anchors(q_vec, limit) if q_vec is not None else []
        ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        for channel, ids in (("lexical_rank", lexical), ("semantic_rank", semantic)):
            for rank, tid in enumerate(ids, 1):
                ranks[tid][channel] = rank
        result = []
        for tid, channels in ranks.items():
            row = self.get_turns([tid])[0]
            row.update(turn_id=tid, score=sum(1 / (60 + r) for r in channels.values()), **channels)
            result.append(row)
        return sorted(result, key=lambda r: (-r["score"], r["turn_id"]))[:limit]

    def _sem_anchors(
        self,
        q_vec: List[float],
        top_k: int,
    ) -> List[str]:
        """Top-``top_k`` turns by cosine(q_vec, emb), over embedded turns only.
        Deterministic; numpy imported lazily so a missing dependency degrades
        to the four-channel path instead of raising.
        """
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("[evidence] numpy unavailable for semantic channel: %s", exc)
            return []
        try:
            q = np.asarray(q_vec, dtype=np.float32)
            qn = float(np.linalg.norm(q))
            if qn == 0:
                return []
            sims = []
            for tid, turn in self._turn_by_id.items():
                emb = turn.get("emb")
                if not emb:
                    continue
                v = np.asarray(emb, dtype=np.float32)
                c = float(q @ v / (qn * float(np.linalg.norm(v)) + 1e-9))
                sims.append((c, tid))
            sims.sort(key=lambda kv: -kv[0])
            return [tid for _, tid in sims[:top_k]]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[evidence] semantic anchor computation failed: %s", exc)
            return []
