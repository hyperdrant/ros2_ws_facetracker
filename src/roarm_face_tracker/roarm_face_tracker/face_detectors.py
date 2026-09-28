"""
人脸检测器：把"检测"这件事从 ROS 节点里独立出来。

两个检测器提供同样的接口：
    detector.detect(frame_bgr) -> [(x, y, w, h, score), ...]   （原图像素坐标）
所以跟踪节点不关心具体用哪种算法，换算法只需要改参数 detector。

- HaarFaceDetector  OpenCV 自带的传统方法。很快、不需要额外依赖，
                    但只认正脸，侧脸、低头、暗光、远距离时容易漏检。
- YoloFaceDetector  深度学习方法（ultralytics YOLO + 人脸模型），
                    对角度、光照、距离都鲁棒得多；在 RTX 3060 上每帧约 10ms。
                    需要 ultralytics + PyTorch（本机系统 Python 已安装，
                    它们不是 ROS 包，所以没有写进 package.xml）。
"""
import cv2


class HaarFaceDetector:
    def __init__(self, detect_width=320, min_face_size=30):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f'无法加载人脸模型 {path}')
        self.detect_width = detect_width
        self.min_face_size = min_face_size

    def detect(self, frame):
        # 缩小图像以提高速度
        h, w = frame.shape[:2]
        scale = self.detect_width / w if w > self.detect_width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 直方图均衡化，减弱光照影响
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # 图像金字塔每层缩小 10%
            minNeighbors=5,    # 越大误检越少，但也更容易漏检
            minSize=(self.min_face_size, self.min_face_size))

        # Haar 不给置信度，统一填 1.0；坐标换算回原图
        return [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale), 1.0)
                for (x, y, fw, fh) in faces]


class YoloFaceDetector:
    def __init__(self, model_path, conf=0.4, device='0', imgsz=640):
        # 在这里才 import：选 haar 时就不必加载 PyTorch（加载要 1~2 秒、占不少内存）
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf        # 置信度阈值：低于它的检测框被丢弃
        self.imgsz = imgsz      # 推理时图像缩放到的尺寸
        # device: '0' 表示第 0 块 GPU，'cpu' 表示用 CPU
        self.device = int(device) if str(device).isdigit() else device

        # 预热：第一次推理要初始化 CUDA、分配显存，大约 1 秒。
        # 放在启动时做掉，避免跟踪开始后第一帧卡顿。
        import numpy as np
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def detect(self, frame):
        # predict 返回一个列表，每张输入图像对应一个 Results；这里只输入一张
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]

        faces = []
        # boxes.xyxy：每个框的 [左上x, 左上y, 右下x, 右下y]（原图像素坐标）
        # boxes.conf：每个框的置信度
        # 它们是 GPU 上的 torch 张量，.cpu().numpy() 转成 numpy 才能在 Python 里用
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, conf):
            faces.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(score)))
        return faces
