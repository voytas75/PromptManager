"""Tests for the application notification centre."""

from __future__ import annotations

import pytest

from core.notifications import NotificationCenter, NotificationLevel, NotificationStatus


def test_track_task_publishes_start_and_success() -> None:
    center = NotificationCenter()
    events = []

    center.subscribe(events.append)

    with center.track_task(
        title="Test task",
        start_message="Starting",
        success_message="Done",
        metadata={"foo": "bar"},
    ):
        pass

    assert len(events) == 2
    assert events[0].status is NotificationStatus.STARTED
    assert events[1].status is NotificationStatus.SUCCEEDED
    assert events[1].message == "Done"
    assert events[1].metadata["foo"] == "bar"


def test_track_task_failure_includes_exception() -> None:
    center = NotificationCenter()
    failure_events = []
    center.subscribe(failure_events.append)

    with pytest.raises(RuntimeError, match="boom"):
        with center.track_task(
            title="Explode",
            start_message="Start",
            success_message="Success",
            failure_message="Failed",
        ):
            raise RuntimeError("boom")

    assert failure_events[-1].status is NotificationStatus.FAILED
    assert "Failed" in failure_events[-1].message
    assert "boom" in failure_events[-1].message
    assert failure_events[-1].level is NotificationLevel.ERROR


def test_subscription_can_be_closed() -> None:
    center = NotificationCenter()
    events = []
    subscription = center.subscribe(events.append)
    subscription.close()

    with center.track_task(
        title="Silent",
        start_message="Start",
        success_message="Done",
    ):
        pass

    assert not events
