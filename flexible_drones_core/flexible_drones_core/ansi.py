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

"""ANSI terminal color and style codes for console output.

Usage — import the module and prefix constants with the module name:

    from flexible_drones_core import ansi

    print(f"{ansi.GREEN}OK{ansi.RESET}")
    self.get_logger().error(f"{ansi.RED}[{self.name}] Failed: {e}{ansi.RESET}")

Or use named imports when a file has many call sites and readability benefits
from keeping them short:

    from flexible_drones_core.ansi import RED, GREEN, RESET        # unprefixed
    from flexible_drones_core.ansi import (
        RED as _RED, RESET as _RESET,
    )  # module-private style

Avoid re-defining these constants per-file. No third-party dependency is
needed; these are plain ANSI escape sequences supported on any Linux/macOS
terminal and under Windows 10+ with virtual terminal processing enabled.
"""

# Base colors (bright variants — readable on both dark and light backgrounds)
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
CYAN = '\033[96m'

# Style
BOLD = '\033[1m'
RESET = '\033[0m'

# Bold + color combinations for prominent status messages
BOLD_RED = '\033[1;91m'
BOLD_GREEN = '\033[1;92m'
BOLD_CYAN = '\033[1;96m'
BOLD_ORANGE = '\033[1;38;5;214m'  # 256-color orange; may not render on all terminals
