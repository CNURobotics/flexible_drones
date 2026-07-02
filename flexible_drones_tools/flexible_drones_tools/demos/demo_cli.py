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

"""Shared CLI helpers for demo entry points."""

import sys


ANSI_RED_BOLD = '\033[1;31m'
ANSI_YELLOW_BOLD = '\033[1;33m'
ANSI_RESET = '\033[0m'


def print_failure_summary(demo_name, namespace, failure, cleanup_errors=None):
    """Print a concise, high-visibility demo failure without a traceback."""
    cleanup_errors = cleanup_errors or []
    print('', file=sys.stderr, flush=True)
    print(
        f'{ANSI_RED_BOLD}WARNING: {demo_name} failed for /{namespace}.{ANSI_RESET}',
        file=sys.stderr,
        flush=True,
    )
    print(
        f'{ANSI_YELLOW_BOLD}{failure}{ANSI_RESET}',
        file=sys.stderr,
        flush=True,
    )
    if cleanup_errors:
        print(
            f'{ANSI_YELLOW_BOLD}Cleanup issues: {"; ".join(cleanup_errors)}{ANSI_RESET}',
            file=sys.stderr,
            flush=True,
        )
    print(
        f'{ANSI_YELLOW_BOLD}Cleanup was attempted: land, then disarm if needed.{ANSI_RESET}',
        file=sys.stderr,
        flush=True,
    )
