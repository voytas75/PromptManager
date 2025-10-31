"""Prompt dataclass serialization and helper coverage tests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from models.prompt_model import (
    Prompt,
    _deserialize_list,
    _deserialize_metadata,
    _ensure_datetime,
    _ensure_uuid,
    _serialize_list,
    _serialize_metadata,
)


def test_helper_functions_cover_edge_cases() -> None:
    sample_uuid = uuid.uuid4()
    assert _ensure_uuid(sample_uuid) is sample_uuid
    assert _ensure_uuid(str(sample_uuid)) == sample_uuid

    naive_dt = datetime(2025, 10, 30, 12, 0, 0)
    ensured = _ensure_datetime(naive_dt)
    assert ensured.tzinfo is not None
    auto_now = _ensure_datetime(None)
    assert auto_now.tzinfo == timezone.utc

    assert _serialize_list(["a", "b"]) == ["a", "b"]
    assert _serialize_list({"a", "b"})  # set coverage
    assert _serialize_list("tag") == ["tag"]

    assert _serialize_metadata(None) is None
    assert _serialize_metadata("text") == "text"
    complex_meta = {"score": 1}
    assert json.loads(_serialize_metadata(complex_meta)) == complex_meta

    assert _deserialize_metadata(None) is None
    assert _deserialize_metadata("null") is None
    assert _deserialize_metadata('{"broken": 1') == '{"broken": 1'
    assert _deserialize_metadata('{"a":1}') == {"a": 1}

    assert _deserialize_list(None) == []
    assert _deserialize_list('["x",1]') == ["x", "1"]
    assert _deserialize_list("plain") == ["plain"]
    assert _deserialize_list([1, "b"]) == ["1", "b"]


def test_prompt_roundtrip_metadata_and_record() -> None:
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Serializer",
        description="Covers metadata",
        category="testing",
        tags=["one", "two"],
        language="en",
        context="Sample context",
        example_input="input",
        example_output="output",
        version="2.0",
        author="tester",
        quality_score=9.0,
        usage_count=5,
        rating_count=3,
        rating_sum=27.0,
        related_prompts=["other"],
        modified_by="ci",
        ext2={"nested": True},
        ext4=[0.1, 0.2],
        ext5={"list": [1, 2]},
    )

    metadata = prompt.to_metadata()
    assert metadata["related_prompts"] == json.dumps(["other"], ensure_ascii=False)
    assert json.loads(metadata["ext2"]) == {"nested": True}

    record = prompt.to_record()
    assert record["ext4"] == [0.1, 0.2]

    record["ext2"] = json.dumps(record["ext2"])
    record["ext4"] = json.dumps(record["ext4"])
    record["ext5"] = json.dumps(record["ext5"])

    reconstructed = Prompt.from_record(record)
    assert reconstructed.id == prompt.id
    assert reconstructed.ext2 == {"nested": True}
    assert reconstructed.ext4 == [0.1, 0.2]
    assert reconstructed.rating_count == 3
    assert reconstructed.rating_sum == pytest.approx(27.0)


def test_prompt_from_chroma_handles_stringified_lists() -> None:
    prompt_id = str(uuid.uuid4())
    chroma_record = {
        "id": prompt_id,
        "document": "ignored",
        "metadata": {
            "name": "From Chroma",
            "description": "desc",
            "category": "cat",
            "tags": json.dumps(["alpha", "beta"]),
            "related_prompts": json.dumps(["linked"]),
            "quality_score": 0.5,
            "usage_count": 3,
            "is_active": True,
            "source": "chromadb",
        },
    }

    prompt = Prompt.from_chroma(chroma_record)
    assert prompt.id == uuid.UUID(prompt_id)
    assert prompt.tags == ["alpha", "beta"]
    assert prompt.related_prompts == ["linked"]
