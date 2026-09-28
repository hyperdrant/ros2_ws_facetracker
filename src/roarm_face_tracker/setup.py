"""
setup.py —— ament_python 包的安装脚本。

colcon build 时会执行它，做三件事：
1. 把 roarm_face_tracker/ 目录下的 Python 模块安装到 install/ 里
2. 把 launch/、config/ 等数据文件拷贝到 install/share/<包名>/ 下
   （这样 launch 文件里才能用 get_package_share_directory 找到它们）
3. 通过 entry_points 生成可执行文件，
   让你可以用 `ros2 run roarm_face_tracker <可执行名>` 启动节点
"""
import os
from glob import glob
from setuptools import setup

package_name = 'roarm_face_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # 让 ament 索引知道这个包存在（ros2 pkg list 能看到）
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装 launch 文件和参数文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # 安装 YOLO 人脸模型
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyperdrant',
    maintainer_email='sas573920@gmail.com',
    description='RoArm-M2 face tracking with ROS 2',
    license='GPL-3.0',
    entry_points={
        'console_scripts': [
            # 格式： 可执行名 = 模块路径:函数名
            'roarm_driver = roarm_face_tracker.roarm_driver_node:main',
            'camera = roarm_face_tracker.camera_node:main',
            'face_tracker = roarm_face_tracker.face_tracker_node:main',
        ],
    },
)
