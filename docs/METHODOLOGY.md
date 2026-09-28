# Methodology

## 1. Dataset Preparation

- **Authentic samples:** Synthetic plates with uniform typography and clean backgrounds
- **Tampered samples:** Same plates with simulated alterations (character replacement, noise injection, patch overlays)
- Generated via `scripts/generate_samples.py`
- For real-world evaluation, collect images from public datasets (e.g., Indian Number Plate Dataset) and label manually

## 2. Plate Localization

Classical CV approach (no training required for demo):
- Edge detection highlights plate boundaries
- Contour filtering by geometric constraints
- Fallback: center-bottom crop for rear-view vehicle images

**Upgrade path:** Fine-tune YOLOv8 on license plate dataset for higher accuracy.

## 3. OCR & Format Validation

- EasyOCR extracts alphanumeric text
- Post-processing removes non-alphanumeric characters
- Regex validation enforces Indian RTO format

## 4. Tampering Detection

### 4.1 Heuristic Signals

| Signal | Method | Rationale |
|--------|--------|-----------|
| Edge density | Canny edge ratio | Tampered regions create extra edges |
| Texture | LBP histogram variance | Altered areas have different micro-texture |
| Alignment | Contour centroid Y-spread | Manual edits misalign characters |
| Color | HSV mean difference (L/R) | Partial replacement changes hue |

### 4.2 Deep Learning

- **Architecture:** 3-layer CNN (TamperCNN) with BatchNorm and dropout
- **Input:** 224×64 RGB plate crop
- **Output:** Binary classification (authentic / tampered)
- **Training:** Cross-entropy loss, Adam optimizer, 12 epochs default

### 4.3 Score Fusion

```
tamper_score = Σ (weight_i × signal_i)
```

Default weights favor CNN (0.24) and alignment (0.20). Threshold: 0.55.

## 5. Evaluation Metrics

For academic reporting, measure:
- **Accuracy, Precision, Recall, F1** on tampered vs authentic
- **OCR Character Error Rate (CER)**
- **Inference latency** (ms per image)

## 6. Limitations

- Synthetic training data may not generalize to all real-world tampering types
- Low-light and motion blur reduce OCR accuracy
- Requires retraining for non-Indian plate formats

## 7. Proposed Improvements

1. Collect real tampered plate dataset from traffic police collaboration
2. Add attention-based CNN (ResNet18 backbone)
3. Real-time video stream processing with frame sampling
4. Explainability via Grad-CAM heatmaps
