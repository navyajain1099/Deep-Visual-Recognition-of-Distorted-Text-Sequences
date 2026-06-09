# Robust Text Sequence Recognition from Distorted Grayscale Images

This repository contains the final notebook and prediction file for the distorted visual sequence pattern recognition challenge.

The task is to recognize ordered character sequences from noisy, blurred, overlapping, occluded, and deformed grayscale images. The solution uses a custom CRNN model trained from scratch: CNN feature extraction, Bidirectional LSTM sequence modeling, and CTC loss.

## Final Submission Files

- `notebooks/notebook_navya_24119034.ipynb`
- `outputs/submission_navya_24119034.csv`

The submission CSV follows the required format:

```csv
image,prediction
test-0.png,QVTQ8A
test-1.png,7PSW9D
```

## Notebook Contents

The notebook includes the complete workflow:

- project overview and requirement coverage
- dataset extraction and loading
- visual inspection of distorted grayscale images
- label distribution and character vocabulary analysis
- preprocessing and augmentation
- custom CNN + Bidirectional LSTM architecture
- CTC loss training
- validation using Character Error Rate
- exact match accuracy for additional evaluation
- train-validation overlap check
- final test prediction and ordered CSV generation

## Model Summary

No pretrained model is used.

The architecture is implemented directly inside the notebook:

```text
Input grayscale image
→ Custom CNN feature extractor
→ Feature map reshaped into a left-to-right sequence
→ Bidirectional LSTM sequence encoder
→ Linear character classifier
→ CTC decoding
```

## Dataset

The dataset is not committed to the repository.

Expected Colab/Drive setup:

```text
Cig_Ai_open_project/
  notebook_navya_24119034.ipynb
  cig_ps.zip
  outputs/
    checkpoints/
      best.pt
    submission_navya_24119034.csv
```

After extraction in Colab, the dataset should appear as:

```text
/content/data/cig_ps/
  train_images/
  test_images/
  train-labels.csv
```

## Reproducibility

Run the notebook in Google Colab with GPU enabled:

```text
Runtime → Change runtime type → T4 GPU
```

The notebook saves the best model checkpoint and final submission file directly to Google Drive under:

```text
Cig_Ai_open_project/outputs/
```

## Output

Final prediction file:

```text
outputs/submission_navya_24119034.csv
```

