import pytest

from gateway.rag.retrievers import MetadataFilter


def test_simple_equality_filter():
    """Test backward-compatible simple equality."""
    filter = MetadataFilter(values={"team": "ai"})

    assert filter.matches({"team": "ai"})
    assert not filter.matches({"team": "platform"})
    assert not filter.matches({})


def test_explicit_eq_operator():
    """Test $eq operator."""
    filter = MetadataFilter(values={"team": {"$eq": "ai"}})

    assert filter.matches({"team": "ai"})
    assert not filter.matches({"team": "platform"})
    assert not filter.matches({})


def test_ne_operator():
    """Test $ne (not equal) operator."""
    filter = MetadataFilter(values={"team": {"$ne": "archived"}})

    assert filter.matches({"team": "ai"})
    assert filter.matches({"team": "platform"})
    assert not filter.matches({"team": "archived"})


def test_lt_operator():
    """Test $lt (less than) operator."""
    filter = MetadataFilter(values={"priority": {"$lt": 5}})

    assert filter.matches({"priority": 3})
    assert filter.matches({"priority": 4})
    assert not filter.matches({"priority": 5})
    assert not filter.matches({"priority": 10})
    assert not filter.matches({})  # Missing field


def test_lte_operator():
    """Test $lte (less than or equal) operator."""
    filter = MetadataFilter(values={"priority": {"$lte": 5}})

    assert filter.matches({"priority": 3})
    assert filter.matches({"priority": 5})
    assert not filter.matches({"priority": 6})


def test_gt_operator():
    """Test $gt (greater than) operator."""
    filter = MetadataFilter(values={"priority": {"$gt": 3}})

    assert filter.matches({"priority": 5})
    assert filter.matches({"priority": 10})
    assert not filter.matches({"priority": 3})
    assert not filter.matches({"priority": 1})
    assert not filter.matches({})


def test_gte_operator():
    """Test $gte (greater than or equal) operator."""
    filter = MetadataFilter(values={"priority": {"$gte": 3}})

    assert filter.matches({"priority": 3})
    assert filter.matches({"priority": 5})
    assert not filter.matches({"priority": 2})


def test_in_operator():
    """Test $in (in list) operator."""
    filter = MetadataFilter(values={"team": {"$in": ["ai", "ml", "data"]}})

    assert filter.matches({"team": "ai"})
    assert filter.matches({"team": "ml"})
    assert filter.matches({"team": "data"})
    assert not filter.matches({"team": "platform"})
    assert not filter.matches({})


def test_nin_operator():
    """Test $nin (not in list) operator."""
    filter = MetadataFilter(values={"status": {"$nin": ["archived", "deleted"]}})

    assert filter.matches({"status": "active"})
    assert filter.matches({"status": "pending"})
    assert not filter.matches({"status": "archived"})
    assert not filter.matches({"status": "deleted"})


def test_multiple_conditions_on_same_field():
    """Test multiple operators on the same field."""
    filter = MetadataFilter(values={
        "priority": {
            "$gte": 3,
            "$lt": 8,
        }
    })

    assert filter.matches({"priority": 3})
    assert filter.matches({"priority": 5})
    assert filter.matches({"priority": 7})
    assert not filter.matches({"priority": 2})  # < 3
    assert not filter.matches({"priority": 8})  # >= 8
    assert not filter.matches({"priority": 10})


def test_multiple_fields():
    """Test filtering on multiple fields."""
    filter = MetadataFilter(values={
        "team": "ai",
        "priority": {"$gte": 3},
    })

    assert filter.matches({"team": "ai", "priority": 5})
    assert filter.matches({"team": "ai", "priority": 3})
    assert not filter.matches({"team": "platform", "priority": 5})
    assert not filter.matches({"team": "ai", "priority": 2})
    assert not filter.matches({"team": "ai"})  # Missing priority


def test_range_query():
    """Test date/number range queries."""
    filter = MetadataFilter(values={
        "created_at": {
            "$gte": "2024-01-01",
            "$lt": "2024-12-31",
        }
    })

    assert filter.matches({"created_at": "2024-06-15"})
    assert filter.matches({"created_at": "2024-01-01"})
    assert not filter.matches({"created_at": "2023-12-31"})
    assert not filter.matches({"created_at": "2025-01-01"})


def test_tag_filtering_with_in():
    """Test tag-based filtering with $in operator."""
    filter = MetadataFilter(values={
        "tags": {"$in": ["python", "ai"]}
    })

    # Note: This tests if the metadata value is IN the filter list
    # For "tags contains any of", you'd need the metadata to be a list
    assert filter.matches({"tags": "python"})
    assert filter.matches({"tags": "ai"})
    assert not filter.matches({"tags": "java"})


def test_empty_filter_matches_all():
    """Test that empty filter matches everything."""
    filter = MetadataFilter(values={})

    assert filter.matches({"team": "ai"})
    assert filter.matches({"priority": 5})
    assert filter.matches({})


def test_unknown_operator_returns_false():
    """Test that unknown operators are treated safely."""
    filter = MetadataFilter(values={
        "team": {"$unknown": "value"}
    })

    # Unknown operators should fail the filter for safety
    assert not filter.matches({"team": "ai"})


def test_mixed_simple_and_operator_filters():
    """Test combining simple equality and operator-based filters."""
    filter = MetadataFilter(values={
        "team": "ai",  # Simple equality
        "priority": {"$gte": 3},  # Operator
        "status": {"$ne": "archived"},  # Operator
    })

    assert filter.matches({
        "team": "ai",
        "priority": 5,
        "status": "active"
    })

    assert not filter.matches({
        "team": "platform",  # Wrong team
        "priority": 5,
        "status": "active"
    })

    assert not filter.matches({
        "team": "ai",
        "priority": 2,  # Priority too low
        "status": "active"
    })

    assert not filter.matches({
        "team": "ai",
        "priority": 5,
        "status": "archived"  # Archived
    })


def test_numeric_comparisons():
    """Test numeric comparison operators."""
    # Integer comparisons
    filter_int = MetadataFilter(values={"count": {"$gt": 100}})
    assert filter_int.matches({"count": 150})
    assert not filter_int.matches({"count": 50})

    # Float comparisons
    filter_float = MetadataFilter(values={"score": {"$gte": 0.95}})
    assert filter_float.matches({"score": 0.98})
    assert filter_float.matches({"score": 0.95})
    assert not filter_float.matches({"score": 0.90})


def test_string_comparisons():
    """Test string comparison operators (lexicographic)."""
    filter = MetadataFilter(values={
        "version": {
            "$gte": "v2.0.0",
            "$lt": "v3.0.0",
        }
    })

    assert filter.matches({"version": "v2.5.0"})
    assert filter.matches({"version": "v2.0.0"})
    assert not filter.matches({"version": "v1.9.0"})
    assert not filter.matches({"version": "v3.0.0"})
