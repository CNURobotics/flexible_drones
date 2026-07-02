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

from types import SimpleNamespace

import numpy as np

from flexible_drones_msgs.msg import DroneStatus as Status
from geometry_msgs.msg import PoseStamped

from flexible_drones_tools.monitoring.drone_logger import DroneLoggerNode


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class _FakeClock:
    def __init__(self, nanoseconds=0):
        self._nanoseconds = nanoseconds

    def now(self):
        return SimpleNamespace(nanoseconds=self._nanoseconds)


def _stamp_msg(msg, nanoseconds):
    msg.header.stamp.sec = nanoseconds // 1_000_000_000
    msg.header.stamp.nanosec = nanoseconds % 1_000_000_000
    return msg


def _status(armed, nanoseconds):
    msg = _stamp_msg(Status(), nanoseconds)
    msg.status_flags = Status.STATUS_ARMED if armed else 0
    msg.battery_current_a = -1.0
    return msg


def _pose(nanoseconds, x=1.0):
    msg = _stamp_msg(PoseStamped(), nanoseconds)
    msg.pose.position.x = x
    msg.pose.orientation.w = 1.0
    return msg


def _make_logger(tmp_path):
    node = DroneLoggerNode.__new__(DroneLoggerNode)
    node.logging_buffer_length = 10
    node.log_path = tmp_path / 'log' / 'drone1'
    node.logging_buffers = [node._new_logging_buffer(), node._new_logging_buffer()]
    node.num_buffers = len(node.logging_buffers)
    node.index_buffer = 0
    node.log_index = 0
    node.log_stamp = 0
    node.logging_start = None
    node._logging_active = False
    node._last_armed = False
    node._write_threads = []
    node._logger = _FakeLogger()
    node._clock = _FakeClock(123)
    node.get_logger = lambda: node._logger
    node.get_clock = lambda: node._clock
    return node


def test_drone_logger_records_only_while_armed(tmp_path):
    node = _make_logger(tmp_path)

    node.pose_cb(_pose(500_000_000))
    assert not node.log_path.exists()
    assert node.logging_buffers[node.index_buffer]['pose_index'] == 0

    node.status_cb(_status(True, 1_000_000_000))
    node.pose_cb(_pose(1_500_000_000, x=2.0))
    node.status_cb(_status(False, 2_000_000_000))

    for thread in node._write_threads:
        thread.join()

    assert not node._logging_active
    assert node.logging_start is None

    pose_file = node.log_path / 'pose_1000000000_0.npy'
    status_file = node.log_path / 'status_1000000000_0.npy'
    assert pose_file.exists()
    assert status_file.exists()

    pose_data = np.load(pose_file)
    status_data = np.load(status_file)
    assert pose_data.shape == (1, 8)
    assert pose_data[0, 0] == 0.5
    assert pose_data[0, 1] == 2.0
    assert status_data.shape == (2, 7)
    assert status_data[0, 0] == 0
    assert status_data[1, 0] == 1_000_000
    assert int(status_data[0, 2]) & Status.STATUS_ARMED
    assert not (int(status_data[1, 2]) & Status.STATUS_ARMED)

    node.pose_cb(_pose(2_500_000_000, x=3.0))
    assert node.logging_buffers[node.index_buffer]['pose_index'] == 0
