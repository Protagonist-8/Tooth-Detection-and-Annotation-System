# post_processing_pipeline.py

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans

# -----------------------------
# PATH VARIABLES (fill these in)
# -----------------------------
MODEL_PATH = ""          # e.g., "/content/drive/MyDrive/.../best.pt"
TEST_IMAGES_PATH = ""    # e.g., "/content/drive/MyDrive/Split_dataset/images/test"

CONFIDENCE = 0.5


def load_model(model_path):
    """Load YOLO model."""
    return YOLO(model_path)


def get_test_image(test_images_path):
    """Pick one test image (first one)."""
    img_file = os.listdir(test_images_path)[0]
    img_path = os.path.join(test_images_path, img_file)
    print(f"Selected Image Path: {img_path}")
    return img_path


def run_predictions(model, img_path, conf=0.5):
    """Run YOLO inference and return list of dicts."""
    results = model.predict(source=img_path, conf=conf, save=False)
    pred_boxes = []
    for det in results[0].boxes:
        xyxy = det.xyxy[0].cpu().numpy().astype(int)  # [x1,y1,x2,y2]
        cls = int(det.cls[0].cpu().numpy())
        pred_boxes.append({"class": cls, "bbox": xyxy})
    return pred_boxes


def separate_upper_lower(predictions):
    """Cluster predictions into upper vs lower arches."""
    for p in predictions:
        x1, y1, x2, y2 = p['bbox']
        p['x_center'] = (x1 + x2) / 2
        p['y_center'] = (y1 + y2) / 2

    y_centers = np.array([p['y_center'] for p in predictions]).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(y_centers)
    labels = kmeans.labels_

    cluster0_mean = y_centers[labels == 0].mean()
    cluster1_mean = y_centers[labels == 1].mean()

    if cluster0_mean < cluster1_mean:
        upper_label, lower_label = 0, 1
    else:
        upper_label, lower_label = 1, 0

    for i, p in enumerate(predictions):
        p['arch'] = 'upper' if labels[i] == upper_label else 'lower'

    return pd.DataFrame(predictions)


def assign_quadrants(df):
    """Divide into left vs right within each arch."""
    df['quadrant'] = None
    for arch in ['upper', 'lower']:
        arch_df = df[df['arch'] == arch]
        if arch_df.empty:
            continue
        x_mid = arch_df['x_center'].median()
        df.loc[(df['arch'] == arch) & (df['x_center'] < x_mid), 'quadrant'] = f"{arch}_left"
        df.loc[(df['arch'] == arch) & (df['x_center'] >= x_mid), 'quadrant'] = f"{arch}_right"
    return df


def assign_fdi_numbers(df):
    """Assign FDI numbers sequentially per quadrant."""
    fdi_map = {
        'upper_right': list(range(11, 19)),
        'upper_left': list(range(21, 29)),
        'lower_left': list(range(31, 39)),
        'lower_right': list(range(41, 49))
    }

    df['fdi_number'] = None

    for quadrant, fdi_numbers in fdi_map.items():
        q_df = df[df['quadrant'] == quadrant].copy()
        if q_df.empty:
            continue
        q_df = q_df.sort_values('x_center', ascending=True).reset_index()
        n_teeth = len(q_df)
        assigned_fdi = fdi_numbers[:n_teeth]
        df.loc[q_df['index'], 'fdi_number'] = assigned_fdi

    return df


def post_process():
    # Load model
    model = load_model(MODEL_PATH)

    # Pick one test image
    img_path = get_test_image(TEST_IMAGES_PATH)
    img = cv2.imread(img_path)

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Predictions
    pred_boxes = run_predictions(model, img_path, CONFIDENCE)
    print("List of bounding boxes predicted:")
    print(pred_boxes)

    # Step 1: Separate Upper vs Lower
    df = separate_upper_lower(pred_boxes)
    print(df[['class', 'x_center', 'y_center', 'arch']])

    # Step 2: Quadrants
    df = assign_quadrants(df)
    print(df[['class', 'arch', 'x_center', 'quadrant']])

    # Step 3: FDI Numbers
    df = assign_fdi_numbers(df)
    print(df[['class', 'arch', 'quadrant', 'x_center', 'fdi_number']])


if __name__ == "__main__":
    main()
