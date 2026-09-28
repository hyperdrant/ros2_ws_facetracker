"""
一键启动：机械臂驱动 + 摄像头 + 人脸跟踪

用法示例：
  ros2 launch roarm_face_tracker face_tracking.launch.py
  ros2 launch roarm_face_tracker face_tracking.launch.py port:=/dev/ttyUSB1

  # 只检测、不控制机械臂（第一次调试时推荐）
  ros2 launch roarm_face_tracker face_tracking.launch.py enable_tracking:=false

  # 不弹出实时画面窗口
  ros2 launch roarm_face_tracker face_tracking.launch.py view:=false

启动后的节点 / 话题结构：

  /camera/camera ──/camera/image_raw──▶ /roarm/face_tracker ──/roarm/joint_command──▶ /roarm/roarm_driver ──串口──▶ 机械臂
                                              ▲                                              │
                                              └──────────────/roarm/joint_states─────────────┘

launch 文件本身也是 Python：generate_launch_description() 返回一个
LaunchDescription，里面列出要启动的节点和"启动参数"(DeclareLaunchArgument)。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # install/share/roarm_face_tracker/config/face_tracking.yaml
    params_file = os.path.join(
        get_package_share_directory('roarm_face_tracker'), 'config', 'face_tracking.yaml')

    # LaunchConfiguration 是"占位符"，真正的值在 launch 运行时才确定
    port = LaunchConfiguration('port')
    enable_tracking = LaunchConfiguration('enable_tracking')
    view = LaunchConfiguration('view')

    # 不控制机械臂时，把跟踪节点的输出指令重定向到一个没人订阅的话题上，
    # 这样它照常检测人脸、发布调试图像，但机械臂不会动。
    cmd_topic = PythonExpression([
        "'joint_command' if '", enable_tracking, "' == 'true' else 'joint_command_dry_run'"])

    return LaunchDescription([
        # ---------- 启动参数（命令行里用 名字:=值 覆盖） ----------
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0',
                              description='机械臂串口'),
        DeclareLaunchArgument('enable_tracking', default_value='true',
                              description='false 时只检测人脸，不控制机械臂'),
        DeclareLaunchArgument('view', default_value='true',
                              description='是否弹出 rqt_image_view 显示实时画面'),

        # ---------- 机械臂驱动 ----------
        Node(
            package='roarm_face_tracker',
            executable='roarm_driver',
            name='roarm_driver',
            namespace='roarm',
            # parameters 列表：先加载 yaml，再用字典覆盖 port（后面的优先级更高）
            parameters=[params_file, {'port': port}],
            output='screen',
        ),

        # ---------- 摄像头 ----------
        Node(
            package='roarm_face_tracker',
            executable='camera',
            name='camera',
            namespace='camera',
            parameters=[params_file],
            output='screen',
        ),

        # ---------- 人脸跟踪 ----------
        Node(
            package='roarm_face_tracker',
            executable='face_tracker',
            name='face_tracker',
            namespace='roarm',
            parameters=[params_file],
            # remappings：把节点里的话题名重定向到别的名字。
            # 节点订阅的是相对名 image_raw（会变成 /roarm/image_raw），
            # 这里把它重定向到摄像头实际发布的 /camera/image_raw。
            # joint_states 不用改，自然就落在 /roarm/ 下，正好对上驱动节点。
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('joint_command', cmd_topic),
            ],
            output='screen',
        ),

        # ---------- 实时画面窗口 ----------
        # 注意：ROS 节点本身不会弹窗，图像只是发布在话题上；
        # rqt_image_view 是一个订阅图像话题并显示出来的 GUI 工具。
        # 它接受一个命令行参数作为默认显示的话题；窗口里的下拉框也可以切换到 /camera/image_raw 看原始画面。
        #
        # prefix：在启动命令前面加一段，实际执行的是
        #   /usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view /roarm/debug_image
        # 为什么需要它？rqt_image_view 的脚本开头是 `#!/usr/bin/env python3`，
        # 会使用 PATH 里的第一个 python3。本机装了 pyenv，找到的是 pyenv 的 Python，
        # 它里面没有 apt 安装的 PyQt5，窗口会直接崩溃。这里强制用系统 Python 运行。
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_view',
            prefix='/usr/bin/python3',
            arguments=['/roarm/debug_image'],
            condition=IfCondition(view),  # 只有 view:=true 时才启动
        ),
    ])
