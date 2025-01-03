from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")



# Train the model on the  dataset dress code  for 100 epochs
results = model.train(data="data.yaml", epochs=100, imgsz=640)

