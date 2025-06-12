from ultralytics import YOLO
model = YOLO('F:/Desktop/YOLO/ultralytics-main/runs/detect/train18/weights/best.pt')
source = 'F:/Desktop/YOLO/ultralytics-main/detect_image' #更改为自己的图片路径
# 运行推理，并附加参数
model.predict(source, save=True)