
from ultralytics import YOLO


MODEL_PATH = ""
DATA_YAML = ""
SAVE_PATH=""

def test_step_YOLO(model_path, data_yaml, save_path, input_size=640, batch_size=16, device=0, workers=1):
    """
    Evaluate YOLO model on test dataset.
    """
    # Load model
    model = YOLO(model_path)

    # Run evaluation
    metrics = model.val(
        data=data_yaml,
        imgsz=input_size,
        batch=batch_size,
        split="test",   
        device=device,
        workers=workers,
        project=save_path,
        name="test_results"
    )

    return metrics


if __name__ == "__main__":
    test_step_YOLO(MODEL_PATH, DATA_YAML, SAVE_PATH)
