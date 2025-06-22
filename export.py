from ultralytics import YOLO
model = YOLO('yolov8m-seg.pt')
model.export(format='onnx')