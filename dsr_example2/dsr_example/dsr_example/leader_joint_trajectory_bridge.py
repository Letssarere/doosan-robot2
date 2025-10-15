"""ROS 2 node that maps external joint trajectories to Doosan servoj_rt commands."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m1013"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class LeaderJointTrajectoryBridge(Node):
    """Bridge `/leader/joint_trajectory` messages to Doosan servoj_rt commands."""

    DOOSAN_JOINT_ORDER: Sequence[str] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )

    def __init__(self) -> None:
        super().__init__("leader_servoj_rt", namespace=ROBOT_ID)
        DR_init.__dsr__node = self

        from DSR_ROBOT2 import ROBOT_MODE_AUTONOMOUS, servoj_rt, set_robot_mode

        self._servoj_rt = servoj_rt
        self._set_robot_mode = set_robot_mode
        self._robot_mode = ROBOT_MODE_AUTONOMOUS
        self._set_robot_mode(self._robot_mode)

        self._subscription = self.create_subscription(
            JointTrajectory,
            "/leader/joint_trajectory",
            self._trajectory_callback,
            10,
        )
        self._last_warn_log_time = self.get_clock().now()

        self.get_logger().info(
            "leader_servoj_rt node is ready to stream trajectories to %s (%s)",
            ROBOT_ID,
            ROBOT_MODEL,
        )

    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return

        point = msg.points[0]
        try:
            positions = self._reorder(msg.joint_names, point.positions, default=None, field_name="positions")
            velocities = self._reorder(msg.joint_names, point.velocities, default=0.0, field_name="velocities")
            accelerations = self._reorder(
                msg.joint_names,
                point.accelerations,
                default=0.0,
                field_name="accelerations",
            )
        except ValueError as err:
            self._warn_throttled(str(err))
            return

        pos_deg = [math.degrees(float(value)) for value in positions]
        vel_deg = [math.degrees(float(value)) for value in velocities]
        acc_deg = [math.degrees(float(value)) for value in accelerations]

        duration = point.time_from_start
        command_time = float(duration.sec) + float(duration.nanosec) * 1.0e-9
        if command_time < 0.0:
            command_time = 0.0

        try:
            self._servoj_rt(pos_deg, vel_deg, acc_deg, command_time)
        except Exception as exc:  # noqa: BLE001 - surface DSR driver errors to the log
            self.get_logger().error("Failed to call servoj_rt: %s", exc)

    def _reorder(
        self,
        joint_names: Sequence[str],
        values: Sequence[float],
        *,
        default: float | None,
        field_name: str,
    ) -> List[float]:
        if not joint_names:
            raise ValueError("Received trajectory without joint names; cannot map to Doosan joints.")

        if not values:
            if default is None:
                raise ValueError(
                    f"Trajectory {field_name} are empty; cannot compute command for Doosan joints."
                )
            name_to_value: Dict[str, float] = {name: float(default) for name in joint_names}
        else:
            if len(values) != len(joint_names):
                raise ValueError(
                    f"Trajectory {field_name} length ({len(values)}) does not match joint_names length ({len(joint_names)})."
                )
            name_to_value = {name: float(value) for name, value in zip(joint_names, values)}

        missing_names = [name for name in self.DOOSAN_JOINT_ORDER if name not in name_to_value]
        if missing_names:
            raise ValueError(
                "Trajectory joint_names are missing required Doosan joints: " + ", ".join(missing_names)
            )

        return [name_to_value[name] for name in self.DOOSAN_JOINT_ORDER]

    def _warn_throttled(self, message: str) -> None:
        now = self.get_clock().now()
        if (now - self._last_warn_log_time).nanoseconds > 1_000_000_000:
            self.get_logger().warn(message)
            self._last_warn_log_time = now


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeaderJointTrajectoryBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
