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

"""Launch RViz for one registry-backed fleet or an explicit heterogeneous flier list."""

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
from flexible_drones_description.launch_assets import make_rviz_launch_actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


DEFAULT_DEPLOYMENT = "mixed"


def _optional_int_launch_value(context, name: str) -> int | None:
    value = LaunchConfiguration(name).perform(context)
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer pixel coordinate, got {value!r}") from exc


def _build_actions(context, *args, **kwargs):
    del args, kwargs

    selected = parse_selected_names(LaunchConfiguration("fliers").perform(context))
    if selected:
        drone_names = selected
    else:
        roster_package = LaunchConfiguration("roster_package").perform(context)
        deployment_package = LaunchConfiguration("deployment_package").perform(context)
        direct_fliers_hint = (
            'To launch RViz without deployment files, pass `fliers:="[cf1, cf2]"`.'
        )
        deployment_package = require_package_name(
            deployment_package,
            argument_name='deployment_package',
            env_var=DEPLOYMENT_PACKAGE_ENV,
            purpose='resolve roster and deployment setup files when fliers is empty',
            extra_hint=direct_fliers_hint,
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
        specs = load_deployment_specs(roster_path, deployment_setup_path)
        if not specs:
            raise RuntimeError(
                f"No drones selected from roster '{roster_path}' and deployment '{deployment_setup_path}'."
            )
        drone_names = [spec.name for spec in specs]

    return make_rviz_launch_actions(
        drone_names,
        LaunchConfiguration("rviz_config").perform(context),
        tempfile_prefix="flexible_drones_description_rviz_",
        window_x=_optional_int_launch_value(context, "window_x"),
        window_y=_optional_int_launch_value(context, "window_y"),
        window_width=_optional_int_launch_value(context, "window_width"),
        window_height=_optional_int_launch_value(context, "window_height"),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fliers",
                default_value="",
                description=(
                    "YAML list of drone names to visualize, e.g. [cf1, px1]. Empty means all "
                    "enabled drones listed in the deployment setup; non-empty directly "
                    "visualizes exactly those names and ignores roster/deployment arguments."
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
            DeclareLaunchArgument(
                "rviz_config",
                default_value="empty.rviz",
                description="Base RViz config filename in flexible_drones_description/rviz, or an absolute path.",
            ),
            DeclareLaunchArgument(
                "window_x",
                default_value="",
                description="Optional RViz window X position in screen pixels.",
            ),
            DeclareLaunchArgument(
                "window_y",
                default_value="",
                description="Optional RViz window Y position in screen pixels.",
            ),
            DeclareLaunchArgument(
                "window_width",
                default_value="",
                description="Optional RViz window width in screen pixels.",
            ),
            DeclareLaunchArgument(
                "window_height",
                default_value="",
                description="Optional RViz window height in screen pixels.",
            ),
            OpaqueFunction(function=_build_actions),
        ]
    )
