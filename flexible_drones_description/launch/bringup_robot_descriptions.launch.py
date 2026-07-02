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

"""Launch shared robot description publishers for a selected fleet."""

from flexible_drones_core.deployment import (
    DEFAULT_DEPLOYMENT_PACKAGE,
    DEFAULT_DEPLOYMENT_SETUP,
    DEFAULT_ROSTER,
    DEPLOYMENT_PACKAGE_ENV,
    load_deployment_specs,
    parse_selected_names,
    require_package_name,
    resolve_deployment_setup_path,
    resolve_roster_path,
)
from flexible_drones_description.launch_assets import (
    make_prop_joint_state_spinner,
    make_robot_state_publisher,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


DEFAULT_DEPLOYMENT = "mixed"


def _build_actions(context, *args, **kwargs):
    del args, kwargs

    roster_package = LaunchConfiguration("roster_package").perform(context)
    deployment_package = LaunchConfiguration("deployment_package").perform(context)
    deployment_package = require_package_name(
        deployment_package,
        argument_name='deployment_package',
        env_var=DEPLOYMENT_PACKAGE_ENV,
        purpose='resolve roster and deployment setup files',
    )
    roster_package = roster_package.strip() or deployment_package.strip()
    roster_path = resolve_roster_path(
        LaunchConfiguration("roster").perform(context),
        package_name=roster_package,
    )
    deployment_setup_path = resolve_deployment_setup_path(
        LaunchConfiguration("deployment").perform(context),
        setup_file=LaunchConfiguration("deployment_setup").perform(context),
        package_name=deployment_package,
    )
    selected = parse_selected_names(LaunchConfiguration("fliers").perform(context))

    specs = load_deployment_specs(
        roster_path,
        deployment_setup_path,
        selected_names=selected,
    )
    if not specs:
        raise RuntimeError(
            f"No drones selected from roster '{roster_path}' and deployment '{deployment_setup_path}'."
        )

    drone_names = [spec.name for spec in specs]
    actions = []
    actions.extend(make_robot_state_publisher(spec) for spec in specs)
    actions.append(make_prop_joint_state_spinner(drone_names))

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fliers",
                default_value="",
                description=(
                    "YAML list of drone names to describe, e.g. [cf1, cf3]. Empty means all "
                    "enabled drones listed in the deployment setup; non-empty means exactly "
                    "that subset."
                ),
            ),
            DeclareLaunchArgument(
                "roster",
                default_value=DEFAULT_ROSTER,
                description="Roster YAML absolute path, relative path, or package-relative path.",
            ),
            DeclareLaunchArgument(
                "roster_package",
                default_value="",
                description="Optional package used to resolve the roster path. Empty uses deployment_package.",
            ),
            DeclareLaunchArgument(
                "deployment",
                default_value=DEFAULT_DEPLOYMENT,
                description="Named deployment folder under deployments/.",
            ),
            DeclareLaunchArgument(
                "deployment_setup",
                default_value=DEFAULT_DEPLOYMENT_SETUP,
                description="Optional explicit deployment setup path or setup filename. Empty means deployments/<deployment>/setup.yaml.",
            ),
            DeclareLaunchArgument(
                "deployment_package",
                default_value=DEFAULT_DEPLOYMENT_PACKAGE,
                description=(
                    "Package used to resolve package-relative deployment setup paths. "
                    f"Env fallback: {DEPLOYMENT_PACKAGE_ENV}."
                ),
            ),
            OpaqueFunction(function=_build_actions),
        ]
    )
