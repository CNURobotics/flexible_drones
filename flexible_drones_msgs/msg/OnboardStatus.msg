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

uint8 AUTOPILOT_UNKNOWN = 0
uint8 AUTOPILOT_ARDUPILOT = 1
uint8 AUTOPILOT_PX4 = 2
uint8 AUTOPILOT_SIM = 3

uint8 STATE_UNKNOWN = 0
uint8 STATE_STARTING = 1
uint8 STATE_RUNNING = 2
uint8 STATE_RECONNECTING = 3
uint8 STATE_DEGRADED = 4
uint8 STATE_EMERGENCY = 5

# Health bits are positive indicators. If a bit is not present in
# applicable_health_flags, the check is not meaningful for this configuration.
uint64 HEALTH_ROS_TIME_SYNCED = 1
uint64 HEALTH_FCU_TIME_SYNCED = 2
uint64 HEALTH_TIME_SYNC_CONVERGED = 4
uint64 HEALTH_ODOM_TIMESTAMP_FROM_FCU = 8
uint64 HEALTH_FCU_EPOCH_VALID = 16
uint64 HEALTH_MOCAP_HEALTHY = 32
uint64 HEALTH_ESTIMATOR_READY = 64
uint64 HEALTH_FCU_PREARM_READY = 128
uint64 HEALTH_LOCAL_REFERENCE_VALID = 256
uint64 HEALTH_HOME_POSITION_VALID = 512
uint64 HEALTH_HEARTBEAT_FRESH = 1024
uint64 HEALTH_FCU_STATUS_NOMINAL = 2048
uint64 HEALTH_COMMAND_INTERFACE_IDLE = 4096
uint64 HEALTH_EXTERNAL_NAV_ACCEPTED = 8192

# Blocker bits are negative indicators. A set bit explains why
# ready_for_actions is false.
uint64 BLOCKER_ONBOARD_NOT_RUNNING = 1
uint64 BLOCKER_HEARTBEAT_STALE = 2
uint64 BLOCKER_TIME_SYNC_NOT_CONVERGED = 4
uint64 BLOCKER_FCU_EPOCH_INVALID = 8
uint64 BLOCKER_MOCAP_UNHEALTHY = 16
uint64 BLOCKER_ESTIMATOR_NOT_READY = 32
uint64 BLOCKER_FCU_PREARM_NOT_READY = 64
uint64 BLOCKER_LOCAL_REFERENCE_INVALID = 128
uint64 BLOCKER_HOME_POSITION_INVALID = 256
uint64 BLOCKER_ACTIVE_COMMAND = 512
uint64 BLOCKER_FCU_DEGRADED = 1024
uint64 BLOCKER_FCU_EMERGENCY = 2048
uint64 BLOCKER_RECOVERY_PENDING = 4096
uint64 BLOCKER_ROS_TIME_NOT_SYNCED = 8192
uint64 BLOCKER_FAILSAFE_ACTIVE = 16384

# Warning bits are abnormal conditions that may not block actions.
uint64 WARNING_STATUSTEXT_ACTIVE = 1
uint64 WARNING_MOCAP_STALE = 2
uint64 WARNING_REALIGN_LOCAL_POSITION_ACTIVE = 4
uint64 WARNING_RECOVERY_ATTEMPTED = 8
uint64 WARNING_HOME_SETUP_PENDING = 16
uint64 WARNING_STREAM_REQUEST_FAILED = 32
uint64 WARNING_FCU_EPOCH_INVALID = 64
uint64 WARNING_FCU_LOAD_HIGH = 128

std_msgs/Header header

string drone_name
uint8 autopilot
uint8 state

bool ready_for_actions
uint64 health_flags
uint64 applicable_health_flags
uint64 blocker_flags
uint64 warning_flags

string status
string warning
string fcu_mode
string fcu_system_status
string fcu_status_detail
string readiness_detail
string recovery_detail

int64 estimated_offset_ns
int64 observed_offset_ns
uint64 remote_timestamp_ns
float32 round_trip_time_ms
float32 fcu_epoch_time_error_ms
uint32 accepted_timesync_samples
uint32 rejected_timesync_samples
float32 mocap_sample_age_ms
float32 heartbeat_age_ms
float32 odom_age_ms

# FCU scheduler mainloop load: SYS_STATUS.load / 10, percent (0-100+).
# Headroom gauge for the autopilot CPU; negative means not yet reported.
float32 fcu_load_percent
