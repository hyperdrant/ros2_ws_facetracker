#!/usr/bin/env python3
"""
USB 摄像头节点：用 OpenCV 读取摄像头，发布 sensor_msgs/Image。

  发布  image_raw  (sensor_msgs/Image, bgr8)

为什么把"采图"和"人脸检测"拆成两个节点？
  这是 ROS 的典型设计思路：每个节点只做一件事，通过话题解耦。
  好处是：可以用 rqt_image_view 单独看摄像头画面；以后换成官方的 usb_cam /
  v4l2_camera 驱动，或者换成回放的 rosbag，人脸跟踪节点完全不用改。

单独运行示例：
  ros2 run roarm_face_tracker camera --ros-args -p device:=/dev/video2
  ros2 run rqt_image_view rqt_image_view      # 另开终端查看画面
"""
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from roarm_face_tracker.image_utils import IMAGE_QOS, cv2_to_imgmsg


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera')

        # 参数：摄像头设备、分辨率、帧率、坐标系名
        # device 是字符串，可以填：
        #   - 设备路径，如 '/dev/video2'
        #   - 固定路径，如 '/dev/v4l/by-id/usb-046d_081b_...-video-index0'（推荐）
        #     /dev/videoN 的编号会随插拔顺序变化，by-id 路径按设备型号+序列号命名，不会变。
        #     用 `ls -l /dev/v4l/by-id/` 查看；每个摄像头通常有 index0/index1 两个，
        #     index0 才是出图像的那个，index1 是元数据接口。
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_link')
        # 画面旋转角度（顺时针，只能是 0 / 90 / 180 / 270）。
        # 摄像头如果是横着或倒着装的，在这里把画面转正，
        # 后面的人脸检测、跟踪控制、调试画面就都不用关心摄像头是怎么装的了。
        # 注意：转 90/270 度后宽高互换，640x480 会变成 480x640（竖屏）。
        self.declare_parameter('rotate', 0)

        device = self.get_parameter('device').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        rotate = self.get_parameter('rotate').value
        rotate_codes = {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        if rotate not in rotate_codes:
            raise ValueError(f'rotate 只能是 0/90/180/270，收到 {rotate}')
        self.rotate_code = rotate_codes[rotate]

        # 用 V4L2 后端打开摄像头（Linux 下最稳定）
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'无法打开摄像头 {device}（设备不存在，或被其他程序占用）')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # 驱动缓冲区设为 2 帧：跟踪要的是"最新"画面，缓冲越多延迟越大；
        # 但不能设成 1：只有一个缓冲区时，程序处理当前帧期间下一帧没地方放会被丢掉，
        # 实测帧率会直接减半（15 fps -> 8 fps）。
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # QoS（服务质量）配置见 image_utils.IMAGE_QOS 的注释。
        # 注意：发布方和订阅方的 QoS 必须兼容，否则收不到消息
        # （例如发布方 BEST_EFFORT、订阅方 RELIABLE 就不兼容），所以两边共用同一个配置。
        self.pub = self.create_publisher(Image, 'image_raw', IMAGE_QOS)

        self.timer = self.create_timer(1.0 / fps, self.capture)
        self.get_logger().info(
            f'摄像头 {device} 已打开，实际分辨率 '
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}，顺时针旋转 {rotate}°')

    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('读取图像失败', throttle_duration_sec=2.0)
            return
        if self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)
        msg = cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C 或被 launch 关闭时正常退出
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
