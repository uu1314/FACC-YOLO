import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    #model = YOLO(model=r'F:\Desktop\YOLO\ultralytics-main\yolov8s.pt')
    #model = YOLO(model=r'F:\Desktop\YOLO\ultralytics-main\ultralytics\cfg\models\v6\yolov6.yaml')
    model = YOLO(model=r'F:\Desktop\YOLO\ultralytics-main\ultralytics\cfg\models\11\yolo11-CBAM.yaml')
    #model = YOLO(model=r'F:\Desktop\YOLO\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')
    model.train(data=r'F:\Desktop\YOLO\ultralytics-main\ultralytics\cfg\datasets\dianluban.yaml',
                #resume=False,
                #project='runs/train',

                #name='exp',
                #single_cls=False,
                cache=False,  # 是否缓存数据集以加快后续训练速度，False表示不缓存
                imgsz=640,  # 指定训练时使用的图像尺寸，640表示将输入图像调整为640x640像素
                epochs=150,  # 设置训练的总轮数为200轮
                batch=16,  # 设置每个训练批次的大小为16，即每次更新模型时使用16张图片
                workers=8,  # 设置用于数据加载的线程数为8，更多线程可以加快数据加载速度
                device='0',  # 指定使用的设备，'0'表示使用第一块GPU进行训练
                optimizer='SGD',  # 设置优化器为SGD（随机梯度下降），用于模型参数更新
                amp=True,
                )
