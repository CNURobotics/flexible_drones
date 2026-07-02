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

"""Publish fake JointState values for continuous prop joints from robot descriptions."""

import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class PropJointStateSpinner(Node):
    """Animate continuous prop joints for RViz visualization."""

    def __init__(self):
        super().__init__('prop_joint_state_spinner')

        self.declare_parameter('drone_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('spin_rate_rad_s', 40.0)

        self._drone_names = [str(name).strip() for name in self.get_parameter('drone_names').value]
        self._publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self._spin_rate_rad_s = float(self.get_parameter('spin_rate_rad_s').value)
        self._joint_names = []
        self._seen_joint_names = set()
        self._description_topics_pending = set(self._drone_names)
        self._angle = 0.0
        self._last_time = self.get_clock().now()

        transient_local_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._robot_description_subs = []
        for drone_name in self._drone_names:
            topic_name = f'/{drone_name}/robot_description'
            self._robot_description_subs.append(
                self.create_subscription(
                    String,
                    topic_name,
                    lambda msg, name=drone_name: self._on_robot_description(name, msg),
                    transient_local_qos,
                )
            )

        self._joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        period_s = 1.0 / self._publish_rate_hz if self._publish_rate_hz > 0.0 else 1.0 / 30.0
        self._timer = self.create_timer(period_s, self._publish_joint_states)
        self._status_timer = self.create_timer(5.0, self._warn_on_missing_descriptions)

        self.get_logger().info(
            f'Waiting for robot_description topics for {len(self._drone_names)} drone(s).'
        )

    def _on_robot_description(self, drone_name: str, msg: String):
        continuous_joint_names = self._parse_continuous_joint_names(drone_name, msg.data)
        for joint_name in continuous_joint_names:
            if joint_name in self._seen_joint_names:
                continue
            self._seen_joint_names.add(joint_name)
            self._joint_names.append(joint_name)

        if drone_name in self._description_topics_pending:
            self._description_topics_pending.remove(drone_name)
            self.get_logger().info(
                f"Loaded {len(continuous_joint_names)} continuous joints from '{drone_name}'."
            )

        if not self._description_topics_pending:
            self._status_timer.cancel()
            self.get_logger().info(
                f'Publishing fake joint states for {len(self._joint_names)} continuous joints at '
                f'{self._publish_rate_hz:.1f} Hz.'
            )

    def _parse_continuous_joint_names(self, drone_name: str, robot_description: str):
        if not robot_description.strip():
            self.get_logger().warning(
                f"Received empty robot_description for '{drone_name}'."
            )
            return []

        try:
            root = ET.fromstring(robot_description)
        except ET.ParseError as exc:
            self.get_logger().warning(
                f"Failed to parse robot_description for '{drone_name}': {exc}"
            )
            return []

        joint_names = []
        for joint in root.findall('joint'):
            if joint.get('type') != 'continuous':
                continue
            joint_name = (joint.get('name') or '').strip()
            if joint_name:
                joint_names.append(joint_name)
        return joint_names

    def _warn_on_missing_descriptions(self):
        if not self._description_topics_pending:
            return

        pending = ', '.join(sorted(self._description_topics_pending))
        self.get_logger().warning(
            f'Still waiting for robot_description topics from: {pending}'
        )

    def _publish_joint_states(self):
        if not self._joint_names:
            return

        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        if dt > 0.0:
            self._angle = math.fmod(self._angle + self._spin_rate_rad_s * dt, 2.0 * math.pi)
        self._last_time = now

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self._joint_names
        msg.position = [self._angle] * len(self._joint_names)
        self._joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PropJointStateSpinner()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, ShutdownException, ROSInterruptException):
        node.get_logger().info('Shutting down prop_joint_state_spinner ...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
