#!/usr/bin/env python3
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

"""Publish static identity transforms map→<drone>/<odom_tf> for a fleet."""

import rclpy
from rclpy.executors import ExternalShutdownException, ShutdownException
from rclpy.exceptions import ROSInterruptException
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class StaticTFPublisher(Node):
    """Broadcast one static identity transform per drone: map → <drone>/<odom_tf>."""

    def __init__(self):
        super().__init__('publish_static_transforms')

        self.declare_parameter('drone_names', ['cf1'])
        self.declare_parameter('odom_tf_names', ['odom'])

        drone_names = self.get_parameter('drone_names').value
        odom_tf_names = self.get_parameter('odom_tf_names').value

        if not drone_names:
            self.get_logger().warning('No drone names provided. Nothing to publish.')
            return

        if len(odom_tf_names) != len(drone_names):
            self.get_logger().warning(
                f'odom_tf_names length ({len(odom_tf_names)}) != drone_names length '
                f'({len(drone_names)}); defaulting all to "odom".'
            )
            odom_tf_names = ['odom'] * len(drone_names)

        broadcaster = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()

        transforms = []
        for drone_name, odom_tf_name in zip(drone_names, odom_tf_names):
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = 'map'
            tf.child_frame_id = f'{drone_name}/{odom_tf_name}'
            tf.transform.rotation.w = 1.0
            transforms.append(tf)

        broadcaster.sendTransform(transforms)
        self.get_logger().info(
            'Published static transforms: '
            + ', '.join(f'map→{n}/{o}' for n, o in zip(drone_names, odom_tf_names))
        )


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = StaticTFPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, ShutdownException, ROSInterruptException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
