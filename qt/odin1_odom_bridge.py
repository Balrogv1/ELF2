#!/usr/bin/env python3
import json
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Odin1OdomBridge(Node):
    def __init__(self):
        super().__init__("odin1_odom_bridge")
        self.create_subscription(Odometry, "/odin1/odometry", self.on_odom, 10)

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        payload = {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z)}
        print(json.dumps(payload, separators=(",", ":")), flush=True)


def main():
    rclpy.init()
    node = Odin1OdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
        raise
