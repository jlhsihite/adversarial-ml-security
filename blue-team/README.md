# Blue Team Task

Implementation of three detection mechanisms to identify anomalous and out-of-distribution inputs before they compromise the ML training pipeline.

## 🛡️ Detection Systems

### 1. One-Class SVM (Shallow Anomaly Detection)
- **Method:** Decision boundary learning in 512D feature space
- **Strengths:** Fast inference, threshold-stable across data distributions
- **Use case:** Production environments with dynamic upload patterns

### 2. Variational Autoencoder (Deep Anomaly Detection)
- **Method:** Reconstruction-based detection in 1024D pixel space
- **Strengths:** Detects pixel-level corruptions invisible to feature-based methods
- **Use case:** Comprehensive anomaly detection with acceptable latency

### 3. Mahalanobis Distance (OOD Detection)
- **Method:** Statistical distance from class-conditional distributions
- **Strengths:** 8.2× lower false positive rate than baseline methods
- **Use case:** Out-of-distribution filtering with high precision

## 📊 Performance Summary

| Detector | Accuracy | AUROC | AUPRC | Runtime (GPU) |
|----------|----------|-------|-------|---------------|
| OCSVM | 92.2% | 96.9% | 94.5% | ~5 min |
| VAE | 94.2% | 96.3% | 92.9% | ~8 min |
| Mahalanobis | 95.1% | 98.2% | 98.7% | ~2 min |

## 🚀 Running Blue Team

### Prerequisites
Ensure these files are in the `../data/` directory:
- `AD_val.pt` - Anomaly detection validation set
- `AD_test.pt` - Anomaly detection test set
- `OOD_val.pt` - Out-of-distribution validation set
- `OOD_test.pt` - Out-of-distribution test set
- `dataset.pt` - Training data
- `model_chkpt.pt` - Pre-trained ResNet-8 model

### Execution

1. **Navigate to blue-team directory**
```bash
cd blue-team
```

2. **Open the notebook**
```bash
jupyter notebook blue-team.ipynb
```

3. **Run all cells sequentially**
- Cells execute top-to-bottom
- Models automatically saved for Red Team evaluation
- Results saved to `../results/` directory

**Runtime:** ~15-20 minutes on GPU

## 📁 Output Files

**Detection Models** (saved to `../data/`):
- `ocsvm_anomaly_detector.pkl` - Trained OCSVM
- `vae_anomaly_detector.pt` - Trained VAE
- `mahalanobis_ood_detector.pt` - Trained Mahalanobis detector

**Predictions** (saved to `../results/`):
- `1.pt` - OCSVM anomaly predictions
- `2.pt` - VAE anomaly predictions
- `3.pt` - Mahalanobis OOD predictions

## 🔬 Implementation Details

### Feature Extraction
- **Shallow detectors:** 512D features from pre-trained ResNet-8 penultimate layer
- **Deep detector:** 1024D raw pixel values (32×32 grayscale flattened)

### Hyperparameters

**OCSVM:**
- `nu = 0.1` (outlier fraction bound)
- `kernel = 'linear'` (interpretability over complexity)

**VAE:**
- Latent dimension: 16D
- Architecture: Fully connected encoder-decoder
- Loss: Reconstruction + KL divergence

**Mahalanobis:**
- Class-conditional Gaussian modeling
- Threshold optimized via Youden's index

## 📈 Key Findings

1. **Complementary Strengths:** OCSVM excels at ranking stability, VAE at pixel-level detection
2. **False Positive Trade-off:** Mahalanobis achieves lowest FP rate but struggles with viewpoint diversity
3. **Production Recommendation:** Ensemble approach combining all three detectors

Detailed analysis available in: `blue-team-report.pdf`

---

**Next Step:** Run Red Team evaluation to test defense robustness against adversarial attacks.
