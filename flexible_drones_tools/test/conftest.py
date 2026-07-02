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

"""Pytest configuration for flexible_drones_tools tests."""

from pathlib import Path


ANSI_ORANGE = '\033[38;5;208m'
ANSI_RESET = '\033[0m'
CASADI_SKIP_MESSAGE = 'Skipping real CasADi tests: python import casadi failed'


def _write_to_terminal(message):
    try:
        Path('/dev/tty').write_text(message + '\n', encoding='utf-8')
        return True
    except OSError:
        return False


def pytest_terminal_summary(terminalreporter):
    """Report when optional CasADi-backed tests were skipped."""
    skipped_reports = terminalreporter.stats.get('skipped', [])
    skipped_casadi_tests = any(
        'real casadi unavailable' in str(report.longrepr).lower()
        for report in skipped_reports
    )
    if skipped_casadi_tests:
        colored_message = f'{ANSI_ORANGE}{CASADI_SKIP_MESSAGE}{ANSI_RESET}'
        if not _write_to_terminal(colored_message):
            terminalreporter.write_line(CASADI_SKIP_MESSAGE)
