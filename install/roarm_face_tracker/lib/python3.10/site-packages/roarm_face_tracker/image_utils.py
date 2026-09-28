"""
sensor_msgs/Image <-> numpy(OpenCV) 的互相转换。

为什么不用 cv_bridge？
  ROS 2 Humble 通过 apt 安装的 cv_bridge 是针对 numpy 1.x 编译的，
  而这台机器上 pip 装的是 numpy 2.x，混用时 cv_bridge 很容易在运行时崩溃。
  Image 消息本质上就是 "宽、高、编码格式 + 一段字节数组"，自己转换只要几行代码，
  顺便也能帮助理解 Image 消息的结构。

sensor_msgs/Image 主要字段：
  header    时间戳 + 坐标系 frame_id
  height    图像高（行数）
  width     图像宽（列数）
  encoding  像素格式，例如 'bgr8'(OpenCV 默认)、'rgb8'、'mono8'
  step      每一行占多少字节（可能因为内存对齐而大于 width*通道数）
  data      所有像素的字节，按行依次排列
"""
import array

import cv2
import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


# 图像话题使用的 QoS（服务质量）配置，发布方和订阅方共用。
#
# ROS 2 常见的传感器 QoS 是 qos_profile_sensor_data（BEST_EFFORT，尽力而为、不重传），
# 但实测在这台机器上它会丢掉一半的帧：640x480 的 BGR 图像每帧约 900KB，
# DDS 要把它拆成几百个 UDP 小包，BEST_EFFORT 下只要丢一个小包，整帧就作废。
#
# 所以这里改用：
#   RELIABLE     可靠传输：丢了的小包会重传，整帧能完整送达
#   KEEP_LAST 1  只保留最新 1 帧：处理不过来时旧帧直接被覆盖，
#                不会因为排队越积越多而导致延迟越来越大（跟踪最怕延迟）
IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def cv2_to_imgmsg(frame: np.ndarray, encoding: str = 'bgr8') -> Image:
    """OpenCV 图像(numpy 数组) -> sensor_msgs/Image。header 由调用者填写。"""
    msg = Image()
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    msg.step = msg.width * channels
    # 性能关键：data 字段类型是 uint8[]，
    # 如果直接赋值 bytes，rclpy 会逐个元素检查类型（640x480x3 = 92 万次），非常慢；
    # 赋值 typecode 为 'B' 的 array.array 时，rclpy 会走快速路径，直接整块拷贝。
    msg.data = array.array('B', np.ascontiguousarray(frame).tobytes())
    return msg


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> BGR 格式的 numpy 数组（支持 bgr8 / rgb8 / mono8）。"""
    channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'不支持的图像编码: {msg.encoding}')

    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    # 去掉每行末尾可能存在的对齐填充字节
    img = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)

    if msg.encoding == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
