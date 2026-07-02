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

std_msgs/Header header        # stamp and world/pose frame id (e.g. "map")
string child_frame_id         # body frame id for twist (e.g. drone name)

# Field validity bits. Bit set means the grouped ROS-frame field is valid.
uint8 VALID_POSITION=1
uint8 VALID_ORIENTATION=2
uint8 VALID_LINEAR_VELOCITY=4
uint8 VALID_ANGULAR_RATE=8

uint8 valid_mask              # Validity mask for pose/twist fields; 0 means unspecified
geometry_msgs/Pose pose       # position and orientation in header.frame_id (world ENU)
geometry_msgs/Twist twist     # linear and angular velocity in child_frame_id (body FLU)
