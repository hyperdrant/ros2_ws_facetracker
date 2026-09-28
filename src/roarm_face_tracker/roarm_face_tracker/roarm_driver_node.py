#!/usr/bin/env python3
"""
RoArm-M2 串口驱动节点（ROS 2 <-> ESP32 串口 JSON 协议 的桥梁）

========================  这个节点做什么  ========================
机械臂的主控是 ESP32，它通过 USB 串口（115200 波特率）接收一行一行的 JSON 指令，
协议定义见 roarm_m2/RoArm-M2_example/json_cmd.h，本节点用到其中两条：

  {"T":102,"base":0,"shoulder":0,"elbow":1.57,"hand":3.14,"spd":0,"acc":10}
      -> 四个关节一起转到指定角度（单位：弧度）
  {"T":105}
      -> 请求反馈，ESP32 会回一行：
         {"T":1051,"x":..,"y":..,"z":..,"b":..,"s":..,"e":..,"t":..,"torB":..,...}
         其中 b/s/e/t 分别是 base/shoulder/elbow/hand 的当前角度（弧度）

本节点把它们包装成标准的 ROS 2 话题：

  订阅  joint_command  (sensor_msgs/JointState)  —— 别的节点想让机械臂动，就往这里发
  发布  joint_states   (sensor_msgs/JointState)  —— 机械臂当前关节角度，周期发布

========================  ROS 2 概念小抄  ========================
- 节点 Node：一个独立的功能单元（一个进程里可以有多个），这里是 "roarm_driver"
- 话题 Topic：发布/订阅式的消息通道，异步、多对多
- 参数 Parameter：节点的可配置项，可以在 launch 文件 / yaml / 命令行里修改
- 命名空间 Namespace：给节点和话题加前缀。本包把机械臂相关节点放在 /roarm 下，
  话题就变成 /roarm/joint_command、/roarm/joint_states。
  代码里话题名写成相对名（不以 / 开头），命名空间会自动加上去。
  好处：以后如果要同时控制多台机械臂，只需换个命名空间再启动一份，代码一行都不用改。

单独运行示例：
  ros2 run roarm_face_tracker roarm_driver --ros-args -p port:=/dev/ttyUSB0 -r __ns:=/roarm
"""

import json
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

import serial  # pyserial


# 关节名称：与 JointState 消息里的 name 字段一一对应
JOINT_NAMES = ['base', 'shoulder', 'elbow', 'hand']

# 固件反馈 JSON 中，各关节角度对应的键名
FEEDBACK_KEYS = {'base': 'b', 'shoulder': 's', 'elbow': 'e', 'hand': 't'}


class RoArmDriver(Node):
    """继承 rclpy 的 Node 类，所有 ROS 2 功能（话题、参数、定时器）都通过 self 调用。"""

    def __init__(self):
        # 调用父类构造函数，并给节点起名。
        # 注意：launch 文件里可以用 name= 覆盖这个名字。
        super().__init__('roarm_driver')

        # ---------------- 1. 声明参数 ----------------
        # declare_parameter(名字, 默认值)：
        #   只有声明过的参数才能被外部设置；默认值同时决定了参数的类型。
        self.declare_parameter('port', '/dev/ttyUSB0')   # 串口设备
        self.declare_parameter('baudrate', 115200)       # 固件里 Serial.begin(115200)
        self.declare_parameter('feedback_rate', 20.0)    # 多少 Hz 查询一次关节角度
        # T:102 里的 spd/acc：舵机速度(步/秒, 4096 步 = 一圈, 0 表示最快) 和加速度
        self.declare_parameter('move_speed', 500)
        self.declare_parameter('move_acc', 10)
        # 启动时是否让机械臂先回到初始姿态（固件 T:100）
        self.declare_parameter('move_init_on_start', False)

        # 读取参数值：get_parameter(名字).value
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value
        feedback_rate = self.get_parameter('feedback_rate').value
        self.move_speed = int(self.get_parameter('move_speed').value)
        self.move_acc = int(self.get_parameter('move_acc').value)

        # ---------------- 2. 打开串口 ----------------
        # 与官方示例 serial_simple_ctrl.py 保持一致：
        # ESP32 开发板上 DTR/RTS 连着复位和 BOOT 引脚，
        # 打开串口后必须把它们拉低(False)，否则 ESP32 可能被复位或进入下载模式。
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.1, dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)
        self.get_logger().info(f'已打开串口 {port} @ {baud}')

        # 写串口用的锁：订阅回调和定时器回调都会写串口，加锁防止两条 JSON 交错
        self.write_lock = threading.Lock()

        # ---------------- 3. 创建发布者 / 订阅者 ----------------
        # create_publisher(消息类型, 话题名, QoS 队列深度)
        self.state_pub = self.create_publisher(JointState, 'joint_states', 10)

        # create_subscription(消息类型, 话题名, 回调函数, QoS 队列深度)
        # 每收到一条消息，rclpy 就会调用一次 self.on_joint_command(msg)
        self.cmd_sub = self.create_subscription(
            JointState, 'joint_command', self.on_joint_command, 10)

        # ---------------- 4. 定时器 ----------------
        # create_timer(周期秒, 回调)：周期性地请求一次关节反馈
        self.feedback_timer = self.create_timer(1.0 / feedback_rate, self.request_feedback)

        # 最近一次收到的关节角（用于：joint_command 只给了部分关节时，其余关节保持不动）
        self.last_positions = None

        # ---------------- 5. 串口读取线程 ----------------
        # 串口 readline() 是阻塞的，不能放在 ROS 回调里（会卡住整个节点），
        # 所以单独开一个后台线程专门读数据。
        # 注意：rclpy 的 publish() 是线程安全的，可以在这个线程里直接调用。
        self.running = True
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()

        if self.get_parameter('move_init_on_start').value:
            self.send_json({'T': 100})

    # ------------------------------------------------------------------
    # 串口发送
    # ------------------------------------------------------------------
    def send_json(self, data: dict):
        """把字典转成一行 JSON 并通过串口发送（固件以换行符 '\\n' 作为一条指令的结束）。"""
        line = json.dumps(data, separators=(',', ':')) + '\n'
        with self.write_lock:
            try:
                self.ser.write(line.encode('utf-8'))
            except serial.SerialException as e:
                self.get_logger().error(f'串口写入失败: {e}')

    # ------------------------------------------------------------------
    # 订阅回调：收到关节目标
    # ------------------------------------------------------------------
    def on_joint_command(self, msg: JointState):
        """
        收到 joint_command 后发送 T:102 指令。

        JointState 消息的约定：
          msg.name[i] 与 msg.position[i] 一一对应，
          可以只包含部分关节，例如 name=['base','elbow'], position=[0.3, 1.2]
        """
        if self.last_positions is None:
            # 还没收到过反馈，不知道其他关节在哪里，为安全起见先不动
            self.get_logger().warn('尚未收到机械臂反馈，忽略本次指令', throttle_duration_sec=2.0)
            return

        # 以当前角度为基础，用消息里给出的关节覆盖
        goal = dict(self.last_positions)
        for name, pos in zip(msg.name, msg.position):
            if name in goal:
                goal[name] = float(pos)
            else:
                self.get_logger().warn(f'未知关节名: {name}', throttle_duration_sec=5.0)

        self.send_json({
            'T': 102,
            'base': round(goal['base'], 4),
            'shoulder': round(goal['shoulder'], 4),
            'elbow': round(goal['elbow'], 4),
            'hand': round(goal['hand'], 4),
            'spd': self.move_speed,
            'acc': self.move_acc,
        })

    # ------------------------------------------------------------------
    # 定时器回调：请求反馈
    # ------------------------------------------------------------------
    def request_feedback(self):
        self.send_json({'T': 105})

    # ------------------------------------------------------------------
    # 后台线程：读取串口，解析反馈并发布 joint_states
    # ------------------------------------------------------------------
    def read_loop(self):
        while self.running and rclpy.ok():
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                self.get_logger().error(f'串口读取失败: {e}')
                break
            if not raw:
                continue  # 超时，没有数据

            text = raw.decode('utf-8', errors='ignore').strip()
            # 固件还会打印一些调试信息，只处理 JSON 行
            if not text.startswith('{'):
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            if data.get('T') != 1051:  # 1051 = 关节反馈
                continue

            positions = {j: float(data[FEEDBACK_KEYS[j]]) for j in JOINT_NAMES}
            self.last_positions = positions

            # 组装并发布 JointState 消息
            msg = JointState()
            # header.stamp：时间戳，用节点时钟的当前时间
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(JOINT_NAMES)
            msg.position = [positions[j] for j in JOINT_NAMES]
            # effort 字段放舵机负载（固件的 torB/torS/torE/torH），调试时有用
            msg.effort = [float(data.get(k, 0.0)) for k in ('torB', 'torS', 'torE', 'torH')]
            if not rclpy.ok():  # 程序正在退出，ROS 上下文已失效，不再发布
                break
            self.state_pub.publish(msg)

    def destroy_node(self):
        """节点销毁时关闭线程和串口。"""
        self.running = False
        self.reader_thread.join(timeout=1.0)
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    # 标准的 rclpy 程序结构：init -> 创建节点 -> spin -> 清理 -> shutdown
    rclpy.init(args=args)
    node = RoArmDriver()
    try:
        # spin：进入事件循环，不断处理订阅回调、定时器回调，直到 Ctrl+C
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
