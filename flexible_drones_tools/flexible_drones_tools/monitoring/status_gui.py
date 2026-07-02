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


# With assist from ChatGPT

import signal
import sys
import time

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
    )
    from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
    from PySide6.QtCore import Qt, QThread, QTimer
except ImportError as exc:
    raise SystemExit(
        'status_gui requires PySide6: python3 -m pip install PySide6\n'
        'See flexible_drones_tools/README.md for optional GUI dependencies.'
    ) from exc

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import ParameterDescriptor

from flexible_drones_msgs.msg import DroneStatus as Status
from flexible_drones_tools.monitoring.cli_help import print_usage_if_requested

STATUS_PUBLISH_PERIOD_S = 1.0
STATUS_STALE_TIMEOUT_S = 5.0 * STATUS_PUBLISH_PERIOD_S
DEFAULT_WINDOW_X = 100
DEFAULT_WINDOW_Y = 100
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


CLI_USAGE = """\
Usage:
  ros2 run flexible_drones_tools status_gui --ros-args \\
    -p fliers:="['drone1', 'drone2']" -p window_x:=100 -p window_y:=100

Common examples:
  ros2 run flexible_drones_tools status_gui --ros-args -p fliers:="['drone1', 'drone2']"
  ros2 run flexible_drones_tools status_gui --ros-args -r __node:=status_gui -p fliers:="['drone1']"

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  fliers (string array, required)
    Drone names to monitor. The GUI subscribes to /<flier>/status for each name.
  window_x (integer, default 100)
    Initial window x position in screen pixels.
  window_y (integer, default 100)
    Initial window y position in screen pixels.
"""


def usage():
    return '\nDrone Status GUI:\n' + CLI_USAGE


STATUS_FLAG_TEXT = {
    Status.STATUS_CONNECTED: 'Connected',
    Status.STATUS_ARMED: 'Armed',
    Status.STATUS_AIRBORNE: 'Airborne',
    Status.STATUS_READY_FOR_COMMANDS: 'Ready',
    Status.STATUS_AUTONOMOUS_CONTROL: 'Autonomous',
    Status.STATUS_DEGRADED: 'Degraded',
    Status.STATUS_LOCKED: 'Locked',
}

STATE_TEXT = {
    Status.STATE_NOT_READY: 'Not ready',
    Status.STATE_PREARM: 'Prearm',
    Status.STATE_ARMED: 'Armed',
    Status.STATE_TAKEOFF: 'Takeoff',
    Status.STATE_HOVER: 'Hover',
    Status.STATE_GO_TO: 'GoTo',
    Status.STATE_TELEOP: 'Teleop',
    Status.STATE_TRAJECTORY: 'Trajectory',
    Status.STATE_LANDING: 'Landing',
    Status.STATE_CRASHED: 'Crashed',
    Status.STATE_EMERGENCY: 'Emergency',
}

CONTROL_STATE_TEXT = {
    Status.STATE_TAKEOFF: 'Takeoff',
    Status.STATE_GO_TO: 'GoTo',
    Status.STATE_TELEOP: 'Teleop',
    Status.STATE_TRAJECTORY: 'Trajectory',
    Status.STATE_LANDING: 'Landing',
}


class StatusNode(Node):
    def __init__(self):
        super().__init__('status_gui_node')

        fliers_param = self.declare_parameter('fliers', [], ParameterDescriptor(dynamic_typing=True))
        fliers = fliers_param.get_parameter_value().string_array_value

        if not fliers or not isinstance(fliers, list):
            print('[ERROR] No fliers specified. Usage:')
            print(usage())
            sys.exit(1)
        window_x_param = self.declare_parameter(
            'window_x',
            DEFAULT_WINDOW_X,
            ParameterDescriptor(description='Initial window x position in screen pixels.'),
        )
        window_y_param = self.declare_parameter(
            'window_y',
            DEFAULT_WINDOW_Y,
            ParameterDescriptor(description='Initial window y position in screen pixels.'),
        )

        print(f"Creating Drone Status GUI Node for '{fliers}'")

        self.fliers = fliers  # List of Drone names
        self.window_x = window_x_param.get_parameter_value().integer_value
        self.window_y = window_y_param.get_parameter_value().integer_value
        self._flier_subs = {}
        self._status_msgs = {}
        self._status_times = {}
        for flier in fliers:
            self._flier_subs[flier] = self.create_subscription(
                Status, f'/{flier}/status',
                lambda msg, flier=flier: self.status_callback(msg, flier), 10)
            self._status_msgs[flier] = None
            self._status_times[flier] = None

    def status_callback(self, msg, flier):
        self._status_msgs[flier] = msg
        self._status_times[flier] = time.monotonic()

    def status_is_fresh(self, flier):
        stamp = self._status_times.get(flier)
        if stamp is None:
            return False
        return time.monotonic() - stamp <= STATUS_STALE_TIMEOUT_S


class ROS2Thread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def quit(self):  # noqa: A003
        self.running = False
        super().quit()

    def run(self):
        try:
            while rclpy.ok() and self.running:
                # Block and wait for work
                while rclpy.spin_once(self.node, timeout_sec=0.001):
                    # Spins until no work processed before timeout
                    pass
                time.sleep(0.005)  # Cooperative yield to reduce CPU
        except Exception as e:
            print(f'[ROS2Thread] Exception: {e}')


class BitCircle(QWidget):
    def __init__(self, is_active=False, known=False):
        super().__init__()
        self.is_active = bool(is_active)
        self.known = bool(known)
        self.setFixedSize(20, 20)  # Small circle size

    def set_state(self, active, known=True):
        active = bool(active)
        known = bool(known)
        if self.is_active == active and self.known == known:
            return
        self.is_active = active
        self.known = known
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw circle
        pen = QPen(QColor(0, 0, 0))  # Black border for the circle
        painter.setPen(pen)
        if not self.known:
            color = QColor(92, 92, 92)
        elif self.is_active:
            color = QColor(0, 255, 0)
        else:
            color = QColor(200, 70, 70)
        painter.setBrush(color)
        painter.drawEllipse(0, 0, self.width(), self.height())


def status_tooltip_text(text):
    parts = [part.strip() for part in text.split(';')]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return text
    return ';\n'.join(parts)


class ElidedStatusLabel(QLabel):
    def __init__(self, text='N/A'):
        super().__init__()
        self._full_text = ''
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.set_status_text(text)

    def set_status_text(self, text):
        text = text or ' '
        if text == self._full_text:
            return False
        self._full_text = text
        self.setToolTip(status_tooltip_text(text))
        self._update_elided_text()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self):
        display_text = self._full_text.replace('\n', ' ')
        width = max(0, self.contentsRect().width())
        elided_text = QFontMetrics(self.font()).elidedText(
            display_text, Qt.ElideRight, width)
        if self.text() != elided_text:
            self.setText(elided_text)


class StatusGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('Drone Status GUI')
        self.setGeometry(
            int(self.node.window_x),
            int(self.node.window_y),
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
        self.setFocusPolicy(Qt.StrongFocus)  # enable keyPressEvent

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel('Name:'))

        for i in range(16):
            val = 2**i
            if val in STATUS_FLAG_TEXT:
                left_layout.addWidget(QLabel(STATUS_FLAG_TEXT[val]))

        left_layout.addWidget(QLabel('Control Modes:'))
        for text in CONTROL_STATE_TEXT.values():
            left_layout.addWidget(QLabel(text))

        left_layout.addWidget(QLabel('State:'))
        left_layout.addWidget(QLabel('Status Text:'))
        left_layout.addWidget(QLabel('Battery (Volts):'))
        left_layout.addWidget(QLabel('Battery (%):'))
        left_layout.addWidget(QLabel('Battery Current (A):'))
        left_layout.addWidget(QLabel('Link Latency:'))

        main_layout.addLayout(left_layout)

        for flier in self.node.fliers:
            flier_layout = QVBoxLayout()
            flier_layout.addWidget(QLabel(flier))

            # Create a vertical array of BitCircles for each generic status flag.
            bit_widgets = {}
            for i in range(16):
                val = 2**i
                if val in STATUS_FLAG_TEXT:
                    circle = BitCircle()
                    flier_layout.addWidget(circle)
                    bit_widgets[val] = circle

            flier_layout.addWidget(QLabel(''))
            control_widgets = {}
            for state in CONTROL_STATE_TEXT:
                circle = BitCircle()
                flier_layout.addWidget(circle)
                control_widgets[state] = circle

            # labels for state, status text, volts, battery percent, and link latency
            status_label = QLabel('N/A')
            text_label = ElidedStatusLabel('N/A')
            volt_label = QLabel('0.0V')
            batt_percent_label = QLabel('N/A')
            current_label = QLabel('N/A')
            latency_label = QLabel('N/A')

            flier_layout.addWidget(status_label)
            flier_layout.addWidget(text_label)
            flier_layout.addWidget(volt_label)
            flier_layout.addWidget(batt_percent_label)
            flier_layout.addWidget(current_label)
            flier_layout.addWidget(latency_label)

            # Save references for updates later
            setattr(self, f'{flier}_bit_widgets', bit_widgets)
            setattr(self, f'{flier}_control_widgets', control_widgets)
            setattr(self, f'{flier}_status_label', status_label)
            setattr(self, f'{flier}_text_label', text_label)
            setattr(self, f'{flier}_volt_label', volt_label)
            setattr(self, f'{flier}_batt_label', batt_percent_label)
            setattr(self, f'{flier}_current_label', current_label)
            setattr(self, f'{flier}_latency_label', latency_label)

            # add layout for drone to main_layout
            main_layout.addLayout(flier_layout)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status_display)
        self.timer.start(50)  # Update display at 20 hz

    def update_status_display(self):

        for flier in self.node.fliers:
            self.process_status_msg(
                flier,
                self.node._status_msgs[flier],
                self.node.status_is_fresh(flier),
            )

    def process_status_msg(self, flier, msg, status_fresh):
        bit_widgets = getattr(self, f'{flier}_bit_widgets')
        control_widgets = getattr(self, f'{flier}_control_widgets')
        if msg is None or not status_fresh:
            for circle in bit_widgets.values():
                circle.set_state(False, False)
            for circle in control_widgets.values():
                circle.set_state(False, False)
            self._set_unknown_status_labels(flier)
            return

        status_flags = int(msg.status_flags)
        bat_volt = msg.battery_voltage_v

        for val, circle in bit_widgets.items():
            active = (status_flags & val) != 0
            circle.set_state(active, True)

        for state, circle in control_widgets.items():
            active = int(msg.state) == state
            circle.set_state(active, True)

        status_label = getattr(self, f'{flier}_status_label')
        volt_label = getattr(self, f'{flier}_volt_label')
        batt_label = getattr(self, f'{flier}_batt_label')
        current_label = getattr(self, f'{flier}_current_label')
        text_label = getattr(self, f'{flier}_text_label')
        latency_label = getattr(self, f'{flier}_latency_label')

        status_str = STATE_TEXT.get(msg.state, f'State {msg.state}')
        if status_str != status_label.text():
            status_label.setText(status_str)

        text = msg.warning_text or msg.status_text or ' '
        text_label.set_status_text(text)

        volt_text = f'{bat_volt:.1f}V'
        if volt_text != volt_label.text():
            volt_label.setText(volt_text)

        if msg.battery_remaining_pct == Status.PERCENT_UNKNOWN:
            batt_text = 'Unknown'
        else:
            batt_text = f'{int(msg.battery_remaining_pct)}%'
        if batt_text != batt_label.text():
            batt_label.setText(batt_text)

        battery_current = float(getattr(msg, 'battery_current_a', -1.0))
        current_text = 'Unknown' if battery_current < 0.0 else f'{battery_current:.3f} A'
        if current_text != current_label.text():
            current_label.setText(current_text)

        latency = float(msg.link_latency_ms)
        latency_text = 'Unknown' if latency < 0.0 else f'{latency:.1f} ms'
        if latency_text != latency_label.text():
            latency_label.setText(latency_text)

    def _set_unknown_status_labels(self, flier):
        labels = {
            f'{flier}_status_label': 'N/A',
            f'{flier}_volt_label': 'N/A',
            f'{flier}_batt_label': 'N/A',
            f'{flier}_current_label': 'N/A',
            f'{flier}_latency_label': 'N/A',
        }
        for attr, text in labels.items():
            label = getattr(self, attr)
            if label.text() != text:
                label.setText(text)
        getattr(self, f'{flier}_text_label').set_status_text('N/A')


def main(args=None):
    if print_usage_if_requested(CLI_USAGE, args):
        return

    rclpy.init(args=args)

    # Parse 'fliers' from ROS parameters

    status_node = StatusNode()    # Start the ROS spinner in a separate thread
    ros_thread = ROS2Thread(status_node)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = StatusGUI(status_node)
    gui.show()
    gui.setFixedSize(gui.size())  # Lock the size after show

    # Gracefully shutdown on Ctrl+C
    def sigint_handler(sig, frame):
        print('Shutting down gracefully...', flush=True)
        ros_thread.quit()   # Stop the ROS thread
        ros_thread.wait()   # Wait for the ROS thread to finish
        app.quit()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        app.exec()
    finally:
        ros_thread.quit()
        ros_thread.wait()  # Ensure the thread has fully stopped
        status_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        app.quit()
    print("We're outta here!")


if __name__ == '__main__':
    main()
