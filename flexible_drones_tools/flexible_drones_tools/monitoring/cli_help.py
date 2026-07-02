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

"""Small helpers for monitoring console-script help text."""

import sys


def print_usage_if_requested(usage, args=None):
    """Print static usage text when ``-h`` or ``--help`` is present."""
    cli_args = list(sys.argv[1:] if args is None else args)
    if any(arg in ('-h', '--help') for arg in cli_args):
        print(usage)
        return True

    return False
