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

"""Offline trajectory shape generators and experimental scripts."""

from pathlib import Path

# Default output location for generated trajectory CSVs. These scripts are run
# from source (not installed as ros2 entry points), so output is dumped next to
# the generators in a git-ignored folder.
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'


def default_output_path(filename):
    """Return ``OUTPUT_DIR/filename``, creating the output folder if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename
