#!/usr/bin/env python3
"""
人脸跟踪节点：找到画面中最大的人脸，控制机械臂让它保持在画面中央。

  订阅  image_raw      (sensor_msgs/Image)       摄像头图像
  订阅  joint_states   (sensor_msgs/JointState)  机械臂当前关节角（来自驱动节点）
  发布  joint_command  (sensor_msgs/JointState)  关节目标（发给驱动节点）
  发布  debug_image    (sensor_msgs/Image)       画了人脸框的调试图像

========================  控制思路  ========================
摄像头装在机械臂末端，是 "eye-in-hand（眼在手上）" 结构。
RoArm-M2 是一条手臂、两段臂（大臂 + 小臂），共 4 个关节：
  base     底座，绕竖直轴转        -> 控制摄像头 左右 (pan)
  shoulder 肩关节，转动大臂        （这里保持不动）
  elbow    肘关节，转动小臂        -> 控制摄像头 上下 (tilt)
  hand     末端关节/夹爪           （摄像头若装在这一节上，可把 tilt_joint 改成 hand）
为什么 tilt 用肘关节而不是肩关节？
  转肘关节只改变小臂的俯仰，摄像头主要是"低头/抬头"；
  转肩关节会把整条手臂连同摄像头一起大幅挪动位置，画面变化更剧烈、更难调。

每来一帧图像：
  1. 检测所有人脸（YOLO 或 Haar，见 face_detectors.py），选面积最大的那个
     （通常就是离摄像头最近的人）
  2. 计算人脸中心相对画面中心的像素误差 (ex, ey)
  3. 利用摄像头视场角 (FOV) 把像素误差换算成角度误差：
        角度误差 ≈ (像素误差 / 半幅宽度) * (FOV / 2)
     这样 kp 的物理意义很清楚：kp=1 表示"一步就把误差全部转过去"，
     实际上取 0.3~0.6，让它分几次慢慢转过去，更平稳、不容易振荡
  4. 比例控制（P 控制）：目标角 = 当前实际角 + 方向符号 * kp * 角度误差
  5. 限幅：单步最大转角、关节软限位

为什么以"当前实际角"为起点，而不是在上一次的目标角上累加？
  最初的版本是在目标角上不断累加（目标角 += 修正量），真机上会冲过头：
  机械臂转动需要时间、图像也有延迟，在机械臂真正转到位之前，
  画面里的人脸看起来还是偏的，于是修正量被一次又一次地累加上去，
  等机械臂追上目标时已经转过头了，人脸直接被甩出画面。
  这种现象在控制理论里叫"积分饱和 / 延迟导致的超调"。
  改成每次从实际角度出发后，目标永远只比实际位置多出"一部分误差"，不会越积越多。
"""
import math
import os
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from roarm_face_tracker.face_detectors import HaarFaceDetector, YoloFaceDetector
from roarm_face_tracker.image_utils import IMAGE_QOS, cv2_to_imgmsg, imgmsg_to_bgr


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FaceTracker(Node):
    def __init__(self):
        super().__init__('face_tracker')

        # ---------------- 参数 ----------------
        # 用哪两个关节做 pan / tilt
        self.declare_parameter('pan_joint', 'base')
        self.declare_parameter('tilt_joint', 'elbow')
        # 方向符号：如果机械臂往反方向跑，把对应的符号改成相反数即可
        #   base  正方向 = 向左转。人脸在画面右边(ex>0) 需要向右转 -> 角度减小 -> 符号 -1
        #   elbow 正方向 = 向下。 人脸在画面上方(ey<0) 需要抬头   -> 角度减小 -> 符号 +1
        # 前提：画面是"正"的（摄像头横装/倒装时，先用摄像头节点的 rotate 参数把画面转正）。
        # 这类方向问题靠推算容易出错，以真机测试为准。
        self.declare_parameter('pan_sign', -1.0)
        self.declare_parameter('tilt_sign', 1.0)
        # 比例增益（无量纲，见文件开头说明）
        self.declare_parameter('kp_pan', 0.4)
        self.declare_parameter('kp_tilt', 0.4)
        # 摄像头水平/垂直视场角（度）。普通 USB 摄像头大约 60°x45°，不准也没关系，会被 kp 吸收
        self.declare_parameter('hfov_deg', 60.0)
        self.declare_parameter('vfov_deg', 45.0)
        # 死区：误差小于画面宽/高的这个比例时不动，避免在中心附近抖动
        self.declare_parameter('deadband', 0.05)
        # 每次控制最多转多少弧度
        self.declare_parameter('max_step', 0.1)
        # 软限位（弧度）
        self.declare_parameter('pan_min', -1.57)
        self.declare_parameter('pan_max', 1.57)
        self.declare_parameter('tilt_min', 0.3)
        self.declare_parameter('tilt_max', 2.6)
        # 控制频率上限（Hz），摄像头 30fps，但没必要每帧都给机械臂发指令
        self.declare_parameter('control_rate', 10.0)
        # 人脸中心的低通滤波系数 (0~1)，越小越平滑但越迟钝
        self.declare_parameter('smoothing', 0.5)
        # 超过这么多秒没检测到人脸，才认为"跟丢了"（偶尔漏检一两帧不影响）
        self.declare_parameter('lost_timeout', 0.5)
        self.declare_parameter('publish_debug_image', True)

        # 检测器选择：'yolo' 或 'haar'
        self.declare_parameter('detector', 'yolo')
        # YOLO 参数。yolo_model 留空 = 使用包里自带的 models/yolov12n-face.pt
        self.declare_parameter('yolo_model', '')
        self.declare_parameter('yolo_conf', 0.4)
        self.declare_parameter('yolo_device', '0')   # '0' = 第一块 GPU，'cpu' = CPU
        # Haar 参数：检测时把图像缩小到这个宽度；忽略太小的人脸（缩小后图像中的像素）
        self.declare_parameter('detect_width', 320)
        self.declare_parameter('min_face_size', 30)

        # 把参数读到成员变量里（为了简单，这里只在启动时读一次）
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.pan_joint, self.tilt_joint = p('pan_joint'), p('tilt_joint')
        self.pan_sign, self.tilt_sign = p('pan_sign'), p('tilt_sign')
        self.kp_pan, self.kp_tilt = p('kp_pan'), p('kp_tilt')
        self.hfov = math.radians(p('hfov_deg'))
        self.vfov = math.radians(p('vfov_deg'))
        self.deadband = p('deadband')
        self.max_step = p('max_step')
        self.pan_limits = (p('pan_min'), p('pan_max'))
        self.tilt_limits = (p('tilt_min'), p('tilt_max'))
        self.min_period = 1.0 / p('control_rate')
        self.smoothing = p('smoothing')
        self.lost_timeout = p('lost_timeout')
        self.publish_debug = p('publish_debug_image')

        # ---------------- 人脸检测器 ----------------
        # 具体实现见 face_detectors.py，两种检测器接口相同
        detector = p('detector')
        if detector == 'yolo':
            model = p('yolo_model')
            if not model:
                # get_package_share_directory：找到本包安装后的 share 目录，
                # 即 install/roarm_face_tracker/share/roarm_face_tracker/
                # （setup.py 的 data_files 把 models/ 拷贝到了这里）
                model = os.path.join(get_package_share_directory('roarm_face_tracker'),
                                     'models', 'yolov12n-face.pt')
            self.get_logger().info(f'加载 YOLO 模型 {model}（首次加载需要几秒）...')
            self.detector = YoloFaceDetector(model, conf=p('yolo_conf'), device=p('yolo_device'))
        elif detector == 'haar':
            self.detector = HaarFaceDetector(p('detect_width'), p('min_face_size'))
        else:
            raise ValueError(f"未知检测器 '{detector}'，可选 'yolo' 或 'haar'")
        self.get_logger().info(f'人脸检测器: {detector}')

        # ---------------- 话题 ----------------
        # 图像订阅用与摄像头节点发布端相同的 QoS（见 image_utils.IMAGE_QOS）
        self.create_subscription(Image, 'image_raw', self.on_image, IMAGE_QOS)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.cmd_pub = self.create_publisher(JointState, 'joint_command', 10)
        self.debug_pub = self.create_publisher(Image, 'debug_image', IMAGE_QOS)

        # ---------------- 状态变量 ----------------
        self.current = {}        # 最新的关节反馈 {关节名: 角度}
        self.filtered_err = None  # 低通滤波后的归一化误差 (ex, ey)
        self.last_cmd_time = 0.0
        self.last_face_time = 0.0
        self.frame_count = 0      # 用于统计实际处理帧率

        # 每 5 秒打印一次处理帧率（定时器不止能控制硬件，也常用来做这类周期性统计）
        self.create_timer(5.0, self.report_fps)

        self.get_logger().info(
            f'人脸跟踪已启动：pan={self.pan_joint}, tilt={self.tilt_joint}，等待关节反馈...')

    # ------------------------------------------------------------------
    def on_joint_states(self, msg: JointState):
        """保存最新的关节反馈（控制时以它为起点）。"""
        if not self.current:
            self.get_logger().info('已收到关节反馈，开始跟踪')
        self.current = dict(zip(msg.name, msg.position))

    def report_fps(self):
        fps = self.frame_count / 5.0
        self.frame_count = 0
        if fps == 0:
            self.get_logger().warn('没有收到图像，检查摄像头节点是否正常')
        else:
            self.get_logger().info(f'图像处理帧率: {fps:.1f} fps')

    # ------------------------------------------------------------------
    @staticmethod
    def largest(faces):
        """按面积 w*h 选出最大的人脸 (x, y, w, h, score)，没有则返回 None。"""
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    # ------------------------------------------------------------------
    def on_image(self, msg: Image):
        """每来一帧图像调用一次：检测 -> 计算误差 -> 发送关节指令。"""
        try:
            frame = imgmsg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return

        h, w = frame.shape[:2]
        faces = self.detector.detect(frame)
        face = self.largest(faces)
        now = time.monotonic()
        self.frame_count += 1

        if face is not None:
            self.last_face_time = now
            x, y, fw, fh, _ = face
            cx, cy = x + fw / 2.0, y + fh / 2.0
            # 归一化误差：范围 [-1, 1]，画面中心为 0，向右/向下为正
            ex = (cx - w / 2.0) / (w / 2.0)
            ey = (cy - h / 2.0) / (h / 2.0)

            # 一阶低通滤波：new = a * 当前值 + (1-a) * 旧值
            if self.filtered_err is None:
                self.filtered_err = (ex, ey)
            else:
                a = self.smoothing
                fx, fy = self.filtered_err
                self.filtered_err = (a * ex + (1 - a) * fx, a * ey + (1 - a) * fy)

            if now - self.last_cmd_time >= self.min_period:
                self.last_cmd_time = now
                self.control(*self.filtered_err)
        else:
            # 偶尔漏检一两帧时什么也不做（机械臂停在原地等待）；
            # 超过 lost_timeout 秒没看到人脸才清空滤波器，下次检测到时重新开始
            if now - self.last_face_time > self.lost_timeout:
                self.filtered_err = None

        if self.publish_debug:
            self.publish_debug_image(msg, frame, faces, face)

    # ------------------------------------------------------------------
    def control(self, ex, ey):
        """P 控制：根据归一化误差更新目标角并发布。"""
        if self.pan_joint not in self.current or self.tilt_joint not in self.current:
            return  # 还没有关节反馈

        # 死区内误差视为 0
        ex = 0.0 if abs(ex) < self.deadband else ex
        ey = 0.0 if abs(ey) < self.deadband else ey
        if ex == 0.0 and ey == 0.0:
            return

        # 归一化误差 -> 角度误差（弧度）
        ang_x = ex * self.hfov / 2.0
        ang_y = ey * self.vfov / 2.0

        # 比例控制 + 单步限幅
        d_pan = clamp(self.pan_sign * self.kp_pan * ang_x, -self.max_step, self.max_step)
        d_tilt = clamp(self.tilt_sign * self.kp_tilt * ang_y, -self.max_step, self.max_step)

        # 以当前实际角度为起点（见文件开头的说明），再做软限位
        pan_target = clamp(self.current[self.pan_joint] + d_pan, *self.pan_limits)
        tilt_target = clamp(self.current[self.tilt_joint] + d_tilt, *self.tilt_limits)

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        # 只给出要动的两个关节，驱动节点会让其他关节保持当前位置
        cmd.name = [self.pan_joint, self.tilt_joint]
        cmd.position = [pan_target, tilt_target]
        self.cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    def publish_debug_image(self, src_msg, frame, faces, face):
        """画出画面中心、死区、所有人脸（灰框）和被跟踪的人脸（绿框），方便调参。"""
        img = frame.copy()
        h, w = img.shape[:2]
        # 画面中心十字
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        # 死区矩形
        dx, dy = int(self.deadband * w / 2), int(self.deadband * h / 2)
        cv2.rectangle(img, (w // 2 - dx, h // 2 - dy), (w // 2 + dx, h // 2 + dy), (255, 255, 0), 1)
        for (x, y, fw, fh, score) in faces:
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (160, 160, 160), 1)
            cv2.putText(img, f'{score:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (160, 160, 160), 1)
        if face is not None:
            x, y, fw, fh, _ = face
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.line(img, (w // 2, h // 2), (x + fw // 2, y + fh // 2), (0, 0, 255), 2)

        out = cv2_to_imgmsg(img, 'bgr8')
        out.header = src_msg.header  # 沿用原图的时间戳和坐标系
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FaceTracker()
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
