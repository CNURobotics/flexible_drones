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

# A single obstacle for trajectory planning, expressed in the planner's
# preferred frame.
#
# `type` selects the obstacle primitive. Only "cylinder" is supported for now:
# an infinite-radius-capped vertical cylinder centered at (x, y).
#
# `height` is optional: a non-positive value (or NaN) means "use the planner's
# current z_max", i.e. a full-height cylinder spanning the flight volume.

string type     # "cylinder"
float64 x
float64 y
float64 radius
float64 height  # <= 0 or NaN -> use z_max
