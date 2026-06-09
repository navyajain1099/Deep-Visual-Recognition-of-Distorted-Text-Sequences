# Robust Text Sequence Recognition from Distorted Grayscale Images

## Overview

This project addresses distorted visual sequence recognition using deep learning. The objective is to predict ordered character sequences from grayscale images containing noise, blur, overlap, deformation, occlusion, and irregular spacing.

The solution implements a custom Convolutional Recurrent Neural Network trained from scratch. The model combines convolutional feature extraction, Bidirectional LSTM sequence modeling, and Connectionist Temporal Classification loss for end-to-end text sequence recognition without character-level bounding boxes.

## Final Artifacts

```text
notebooks/notebook_navya_24119034.ipynb
outputs/submission_navya_24119034.csv
```

## Submission Format

The generated prediction file follows the required CSV format:

```csv
image,prediction
test-0.png,QVTQ8A
test-1.png,7PSW9D
```

The final CSV contains predictions for all test images in numeric filename order from `test-0.png` to `test-4999.png`.

## Methodology

The complete workflow is implemented in a single notebook and includes:

- dataset extraction and loading
- grayscale image preprocessing
- visual inspection of sample distortions
- label length and character vocabulary analysis
- data augmentation for training robustness
- custom CNN feature extractor
- Bidirectional LSTM sequence encoder
- CTC loss based training
- validation using Character Error Rate
- exact match accuracy reporting
- train-validation overlap verification
- ordered test prediction generation

## Model Architecture

No pretrained models are used.

The architecture is implemented directly in the notebook:

```text
Input grayscale image
-> Custom CNN feature extractor
-> Feature map reshaped into a left-to-right sequence
-> Bidirectional LSTM encoder
-> Linear character classifier
-> CTC decoding
```

The CNN learns spatial features from distorted characters, while the Bidirectional LSTM models sequential dependencies across the image width. CTC loss enables training with complete text labels without requiring character segmentation.

## Dataset Structure

The dataset is not included in the repository. During execution, the dataset is expected to be available as a zip file in Google Drive and extracted in Colab.

Expected extracted structure:

```text
/content/data/cig_ps/
  train_images/
  test_images/
  train-labels.csv
```

Expected Google Drive structure:

```text
Cig_Ai_open_project/
  notebook_navya_24119034.ipynb
  cig_ps.zip
  outputs/
    checkpoints/
      best.pt
    submission_navya_24119034.csv
```

## Execution Environment

The notebook is designed to run on Google Colab with GPU acceleration.

Recommended runtime:

```text
Python 3
T4 GPU
```

The notebook saves model checkpoints and the final prediction file directly to Google Drive under:

```text
Cig_Ai_open_project/outputs/
```

## Evaluation

Validation performance is measured using Character Error Rate. CER is based on Levenshtein distance and measures character-level insertions, deletions, and substitutions between predicted and target strings.

Exact match accuracy is also reported to provide sequence-level evaluation, where a prediction is counted as correct only if the entire predicted string matches the target.

## Output

Final prediction file:

```text
outputs/submission_navya_24119034.csv
```
