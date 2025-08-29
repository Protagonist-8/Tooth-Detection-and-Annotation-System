# Dental YOLOv8 Tooth Detection

This repository contains code for training and post-processing a YOLOv8
model to detect and label teeth using FDI numbering.

## Environment Setup

Recommend using **Conda** for environment management.

```bash
# Create a new environment
conda create -n dental-yolo python=3.10 -y

# Activate environment
conda activate dental-yolo

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

    ├── split_dataset.py            # To split dataset into train,test,val
    ├── train.py                    # Training script
    ├── evaluate.py                 # Evaluation script
    ├── testing.py                  # Testing on test dataset
    ├── post_processing_pipeline.py # Post-processing logic for anatomical correctness
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project documentation
    └── data.yaml                   # Dataset configuration file

## Training

Run the following command to train YOLOv8 on your dataset:

```bash
python train.py
```

## Evaluation

Evaluate the trained model on the test dataset:

```bash
python evaluate.py
```

## Testing

Run inference and generate results on the test set:

```bash
python testing.py
```

## Post-Processing

After predictions, apply the anatomical correction pipeline:

```bash
python post_processing_pipeline.py
```

## 📌 Notes

- Make sure `data.yaml` is correctly configured with dataset paths.\
- Update the paths inside `post_processing_pipeline.py` before
  running.\
- The FDI numbering system is applied during post-processing.
