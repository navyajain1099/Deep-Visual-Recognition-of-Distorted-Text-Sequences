# Distorted Visual Sequence Pattern Recognition

End-to-end deep learning solution for recognizing distorted grayscale text sequences from images.
The project uses a CRNN architecture: CNN feature extractor + BiLSTM sequence encoder + CTC loss.

## What This Satisfies

- Handles noisy, blurred, occluded, overlapping, and irregularly spaced grayscale sequences.
- Trains from image-label pairs without character-level bounding boxes.
- Evaluates validation performance with Character Error Rate (CER).
- Generates the required submission CSV:

```csv
image,prediction
test-0.png,AXU323
```

## Expected Dataset Layout

Place the downloaded dataset in `data/` using one of these common layouts:

```text
data/
  train/
    train-0.png
    train-1.png
  test/
    test-0.png
    test-1.png
  train_labels.csv
```

or:

```text
data/
  train_images/
  test_images/
  labels.csv
```

The label CSV should contain image filenames and labels. Supported column names:

- image: `image`, `filename`, `file`, `id`, `path`
- label: `label`, `text`, `prediction`, `target`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m src.train --data-dir data --epochs 35 --batch-size 64 --img-height 64 --img-width 256
```

The best checkpoint is saved to `outputs/checkpoints/best.pt`.

## Predict

```powershell
python -m src.predict --data-dir data --checkpoint outputs/checkpoints/best.pt --out outputs/submission.csv
```

Rename the final file according to the problem statement:

```text
submission_<name>_<enroll no.>.csv
```

## Notebook

Open `notebooks/CIG_Distorted_Sequence_CRNN_CTC.ipynb` for the full documented workflow:

- problem understanding
- data checks
- preprocessing and augmentation
- model architecture
- training strategy
- CER validation
- test prediction and CSV generation

