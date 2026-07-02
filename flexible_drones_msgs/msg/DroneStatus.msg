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

# Platform-neutral drone status snapshot.

uint8 PLATFORM_UNKNOWN = 0
uint8 PLATFORM_CRAZYFLIE = 1
uint8 PLATFORM_PIHAWK = 2
uint8 PLATFORM_GAZEBO = 3

uint8 STATE_NOT_READY = 0
uint8 STATE_PREARM = 1
uint8 STATE_ARMED = 2
uint8 STATE_TAKEOFF = 3
uint8 STATE_HOVER = 4
uint8 STATE_GO_TO = 5
uint8 STATE_TELEOP = 6
uint8 STATE_TRAJECTORY = 7
uint8 STATE_LANDING = 8
uint8 STATE_CRASHED = 9
uint8 STATE_EMERGENCY = 10

uint16 STATUS_CONNECTED = 1
uint16 STATUS_ARMED = 2
uint16 STATUS_AIRBORNE = 4
uint16 STATUS_READY_FOR_COMMANDS = 8
uint16 STATUS_AUTONOMOUS_CONTROL = 16
uint16 STATUS_DEGRADED = 32
uint16 STATUS_LOCKED = 64

uint8 PERCENT_UNKNOWN = 255

std_msgs/Header header

string drone_name
uint8 platform
uint8 state
uint16 status_flags
uint32 vendor_status_flags

float32 battery_voltage_v
uint8 battery_remaining_pct
float32 battery_current_a

uint8 link_quality_pct
float32 link_latency_ms
float32 odom_age_ms
float32 time_sync_latency_ms

string status_text
string warning_text
