"""Lossless large candidate fields, persisted deltas and legacy row compatibility."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

from openviking.storage.vectordb.store.bytes_row import _PyBytesRow, _PyFieldType, _PySchema
from openviking.storage.vectordb.store.data import CandidateData, DeltaRecord
from openviking.storage.vectordb.store.serializable import serializable
from openviking.storage.vectordb.store.store_manager import StoreManager, create_store_manager
from openviking.storage.vectordb_adapters.local_adapter import LocalCollectionAdapter


@serializable
@dataclass
class LegacyCandidate:
    label: int = 0
    vector: List[float] = field(default_factory=list)
    sparse_raw_terms: List[str] = field(default_factory=list)
    sparse_values: List[float] = field(default_factory=list)
    fields: str = ""
    expire_ns_ts: int = 0


@serializable
@dataclass
class LegacyDelta:
    type: int = 0
    label: int = 0
    vector: List[float] = field(default_factory=list)
    sparse_raw_terms: List[str] = field(default_factory=list)
    sparse_values: List[float] = field(default_factory=list)
    fields: str = ""
    old_fields: str = ""


@pytest.mark.parametrize("size", [0, 65535, 65536, 193240])
def test_candidate_fields_roundtrip_and_single_field_reads(size: int) -> None:
    content = "a" * size
    original = CandidateData(label=17, fields=content, vector=[0.25], expire_ns_ts=99)
    encoded = original.serialize()
    decoded = CandidateData.from_bytes(encoded)
    assert decoded.fields == content
    assert decoded.label == 17
    assert decoded.expire_ns_ts == 99
    assert CandidateData.bytes_row.deserialize_field(encoded, "fields") == content
    assert original.fields == content


def test_chinese_json_and_old_new_delta_fields_are_not_truncated() -> None:
    content = json.dumps({"content": "订单核对完成🌏" * 15000}, ensure_ascii=False)
    delta = DeltaRecord(label=1, fields=content, old_fields=content + "previous")
    encoded = DeltaRecord.serialize_list([delta])[0]
    restored = DeltaRecord.from_bytes(encoded)
    assert restored.fields == content
    assert restored.old_fields == content + "previous"
    assert json.loads(restored.fields)["content"] == "订单核对完成🌏" * 15000


def test_existing_short_candidate_rows_remain_readable() -> None:
    old = LegacyCandidate(label=9, fields='{"content":"before-upgrade"}', expire_ns_ts=23)
    encoded = old.serialize()
    restored = CandidateData.from_bytes(encoded)
    assert restored.fields == old.fields
    assert restored.label == 9
    assert restored.expire_ns_ts == 23
    assert CandidateData.bytes_row.deserialize_field(encoded, "fields") == old.fields


def test_existing_delta_rows_remain_readable() -> None:
    old = LegacyDelta(label=9, fields="new", old_fields="previous")
    restored = DeltaRecord.from_bytes(old.serialize())
    assert restored.fields == "new"
    assert restored.old_fields == "previous"
    assert restored.label == 9


def test_upstream_text_format_matches_all_python_and_native_codecs() -> None:
    from openviking.storage.vectordb import engine
    from openviking.storage.vectordb.engine import _python_api

    py_schema = _PySchema(
        [
            {
                "name": "fields",
                "data_type": _PyFieldType.text,
                "legacy_data_type": _PyFieldType.string,
                "id": 0,
            }
        ]
    )
    fields = [
        {
            "name": "fields",
            "data_type": engine.FieldType.text,
            "legacy_data_type": engine.FieldType.string,
            "id": 0,
        }
    ]
    codecs = [
        _PyBytesRow(py_schema),
        engine.BytesRow(engine.Schema(fields)),
        _python_api.BytesRow(_python_api.Schema(fields)),
    ]
    original = {"fields": "完整内容🌏" * 20000}
    encodings = [codec.serialize(original) for codec in codecs]
    assert len(set(encodings)) == 1
    assert encodings[0].startswith(b"\x00\x01")
    for codec in codecs:
        assert codec.deserialize(encodings[0]) == original
        assert codec.deserialize_field(encodings[0], "fields") == original["fields"]


def test_large_fields_survive_restart_update_and_delete(tmp_path) -> None:
    path = str(tmp_path / "rows")
    content = json.dumps({"content": "订单核对完成" * 15000}, ensure_ascii=False)
    manager = create_store_manager("local", path)
    manager.add_cands_data([CandidateData(label=5, fields=content)])
    del manager
    manager = create_store_manager("local", path)
    assert manager.fetch_cands_data([5])[0].fields == content
    delta = manager.add_cands_data([CandidateData(label=5, fields="updated")])[0]
    assert delta.old_fields == content
    assert delta.fields == "updated"
    delta = manager.delete_data([5])[0]
    assert delta.old_fields == "updated"
    assert manager.fetch_cands_data([5]) == [None]


def test_existing_collection_without_default_index_rebuilds_from_candidates(tmp_path) -> None:
    schema = {
        "CollectionName": "memory",
        "Fields": [
            {"FieldName": "id", "FieldType": "int64", "IsPrimaryKey": True},
            {"FieldName": "vector", "FieldType": "vector", "Dim": 4},
            {"FieldName": "account_id", "FieldType": "string"},
            {"FieldName": "abstract", "FieldType": "string"},
        ],
        "ScalarIndex": ["account_id"],
    }
    adapter = LocalCollectionAdapter("memory", str(tmp_path), "default")
    assert adapter.create_collection(
        "memory", schema, distance="cosine", sparse_weight=0, index_name="default"
    )
    content = "订单核对完成" * 15000
    adapter.get_collection().upsert_data(
        [{"id": 1, "vector": [1.0, 0, 0, 0], "account_id": "a", "abstract": content}]
    )
    adapter.close()
    shutil.rmtree(tmp_path / "memory/index")
    recovered = LocalCollectionAdapter("memory", str(tmp_path), "default")
    try:
        assert not recovered.create_collection(
            "memory", schema, distance="cosine", sparse_weight=0, index_name="default"
        )
        assert recovered.get_collection().has_index("default")
        result = recovered.get_collection().search_by_vector(
            "default",
            dense_vector=[1.0, 0, 0, 0],
            limit=10,
            filters={"op": "must", "field": "account_id", "conds": ["a"]},
            output_fields=["abstract"],
        )
        assert [item.id for item in result.data] == [1]
        assert result.data[0].fields["abstract"] == content
    finally:
        recovered.close()


def test_failed_native_store_initialization_leaves_metadata_and_recovers_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.storage.vectordb.collection import local_collection
    from openviking.storage.vectordb_adapters.base import CollectionAdapter

    schema = {
        "CollectionName": "memory",
        "Fields": [
            {"FieldName": "id", "FieldType": "int64", "IsPrimaryKey": True},
            {"FieldName": "vector", "FieldType": "vector", "Dim": 4},
        ],
    }
    options = {"distance": "cosine", "sparse_weight": 0, "index_name": "default"}
    original_store_factory = local_collection.create_store_manager

    def unavailable_native_store(storage_type: str, path: str = "") -> StoreManager:
        raise ImportError(
            "Native engine backend is missing from this wheel. Missing symbol: PersistStore"
        )

    monkeypatch.setattr(local_collection, "create_store_manager", unavailable_native_store)
    first = LocalCollectionAdapter("memory", str(tmp_path), "default")
    with pytest.raises(ImportError, match="PersistStore"):
        CollectionAdapter.create_collection(first, "memory", schema, **options)
    assert (tmp_path / "memory/collection_meta.json").is_file()
    assert not (tmp_path / "memory/index/default").exists()

    monkeypatch.setattr(local_collection, "create_store_manager", original_store_factory)
    legacy = LocalCollectionAdapter("memory", str(tmp_path), "default")
    try:
        # Reproduce the original restart: metadata makes collection_exists true,
        # so the base adapter exits without creating a queryable index.
        assert not CollectionAdapter.create_collection(legacy, "memory", schema, **options)
        assert legacy.get_collection().list_indexes() == []
        legacy.get_collection().upsert_data([{"id": 1, "vector": [1.0, 0, 0, 0]}])
        assert [item.id for item in legacy.get_collection().fetch_data([1]).items] == [1]
    finally:
        legacy.close()

    fixed = LocalCollectionAdapter("memory", str(tmp_path), "default")
    try:
        assert not fixed.create_collection("memory", schema, **options)
        assert fixed.get_collection().has_index("default")
        result = fixed.get_collection().search_by_vector(
            "default", dense_vector=[1.0, 0, 0, 0], limit=1
        )
        assert [item.id for item in result.data] == [1]
    finally:
        fixed.close()
