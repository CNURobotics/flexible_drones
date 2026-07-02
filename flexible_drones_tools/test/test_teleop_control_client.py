# Copyright 2026 Christopher Newport University
# Capable Humanitarian Robotics and Intelligent Systems Lab (CHRISLAB)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the teleop_control CLI client."""

from threading import Event

from flexible_drones_tools.clients.teleop_control_client import (
    TeleOpControlClient,
)


class _FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


class _FakeFuture:
    def __init__(self, result):
        self._result = result
        self.done_callback = None

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        self.done_callback = callback


class _FakeGoalHandle:
    def __init__(self, *, accepted=True):
        self.accepted = accepted
        self.cancel_count = 0
        self.result_future = _FakeFuture(None)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        return _FakeFuture(None)


def _client():
    client = TeleOpControlClient.__new__(TeleOpControlClient)
    client._goal_handle = None
    client._done = Event()
    client._cancel_requested = False
    client._cancel_after_accept = False
    client._logger = _FakeLogger()
    client.get_logger = lambda: client._logger
    return client


def test_cancel_before_goal_response_cancels_after_acceptance():
    """Remember an early cancel and issue it as soon as the goal is accepted."""
    client = _client()
    goal_handle = _FakeGoalHandle()

    client.cancel()
    client.goal_response_callback(_FakeFuture(goal_handle))

    assert goal_handle.cancel_count == 1
    assert client._cancel_requested is True
    assert client._cancel_after_accept is False
    assert not client._done.is_set()


def test_cancel_before_rejected_goal_marks_done_without_cancel():
    """A pending cancel should not leak after the goal is rejected."""
    client = _client()
    goal_handle = _FakeGoalHandle(accepted=False)

    client.cancel()
    client.goal_response_callback(_FakeFuture(goal_handle))

    assert goal_handle.cancel_count == 0
    assert client._cancel_requested is False
    assert client._cancel_after_accept is False
    assert client._done.is_set()


def test_repeated_pending_cancel_logs_once():
    """Repeated Enter presses before acceptance keep a single pending cancel."""
    client = _client()

    client.cancel()
    client.cancel()

    assert client._cancel_after_accept is True
    assert len(client._logger.info_messages) == 1
