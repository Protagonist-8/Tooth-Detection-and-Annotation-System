from ultralytics import YOLO

EPOCHS=5
BATCH_SIZE=16
INPUT_SIZE=640

def train_step_yolo(yaml_path,model_weights,input_size=640,epochs=5,batch_size=16):
    model=YOLO(model_weights)

    results=model.train(
        data=yaml_path,
        imgsz=input_size,
        epochs=epochs,
        batch=batch_size,
        workers=1,
        device='cpu'
    )

    print("Training is completed with pretrained weights")
    return results

if __name__=="__main__":
    yaml_path='data.yaml'
    model_weights='yolov8n.pt'
    train_step_yolo(
        yaml_path=yaml_path,
        model_weights=model_weights,
        input_size=INPUT_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )