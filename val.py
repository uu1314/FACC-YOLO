import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')  #修改为自己训练的模型路径
    model.val(data='hat.yaml',  #修改为自己的数据集yaml文件
              split='test',
              imgsz=640,
              batch=16,
              iou=0.6,  #阈值可以改，mAP50为0.5的情况下
              conf=0.001,
              workers=8,
              )