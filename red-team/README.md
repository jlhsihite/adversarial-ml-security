# Red Team: Adversarial Attack Assessment

Offensive security testing using Projected Gradient Descent (PGD) attacks to identify vulnerabilities in ML defense systems and quantify adversarial robustness.

## ⚔️ Attack Methodology

### Projected Gradient Descent (PGD)

**Attack Strategy:**
- **L2-norm constraints** for imperceptible perturbations
- **Untargeted attacks:** Maximize misclassification (any wrong class)
- **Targeted attacks:** Force specific misclassification to Class 5
- **Iterative optimization** with gradient normalization and projection

**Configuration:**
- Step sizes (α): 10⁻⁵ to 10¹ (logarithmic spacing)
- Epsilon (ε): 12.0 (optimized for stealth vs effectiveness)
- Max iterations: 100
- Early stopping: Every 20 iterations

## 📊 Attack Performance

| Step Size (α) | Untargeted ASR | Targeted ASR | Mean L2 Norm | Model Accuracy Drop |
|---------------|----------------|--------------|--------------|---------------------|
| 10⁻⁵ | 1.7% | 0.1% | 22.350 | 1.6% |
| 10⁻³ | 21.4% | 0.7% | 22.431 | 19.8% |
| 10⁻¹ | 88.0% | 28.0% | 22.547 | 81.2% |
| 10⁰ | 97.1% | 46.9% | 22.582 | 89.5% |
| 10¹ | **99.4%** | **56.6%** | 23.202 | **91.3%** |

**Key Findings:**
- Attack success depends on optimization trajectory, not just perturbation magnitude
- Targeted attacks ~42% harder than untargeted (stricter constraints)
- Visual imperceptibility maintained across all step sizes

## 🔓 Defense Evasion Results

### Critical Vulnerabilities Discovered

| Defense System | Evasion Rate (α=10⁻¹) | Evasion Rate (α=10¹) | Vulnerability Assessment |
|----------------|----------------------|---------------------|--------------------------|
| **VAE** | 5.6% | **3.0%** | ✅ Most Robust |
| **OCSVM** | 26.5% | 8.2% | ⚠️ Unstable at extremes |
| **Mahalanobis** | 28.4% | **46.4%** | ❌ Inverse robustness problem |

### Exploitation Scenario

**Critical Security Gap:** Adversarial samples successfully evade multi-layer defense to:

1. **Bypass OOD detection** (46.4% evasion on Mahalanobis at high perturbation)
2. **Evade anomaly detection** (48.3% evasion on OCSVM at α=10⁰)
3. **Maintain visual imperceptibility** (L2 norms ~22-23, indistinguishable to humans)
4. **Contaminate training pipeline** with mislabeled examples
5. **Cause progressive model degradation** over retraining cycles

**Real-world Impact:** Adversaries can inject malicious samples that appear legitimate but systematically degrade classification performance.

## 🚀 Running Red Team

### Prerequisites

**Must complete Blue Team first!** Required files in `../data/`:
- `dataset.pt` - Training data
- `model_chkpt.pt` - Pre-trained ResNet-8 model
- `ocsvm_anomaly_detector.pkl` - Blue Team OCSVM
- `vae_anomaly_detector.pt` - Blue Team VAE
- `mahalanobis_ood_detector.pt` - Blue Team Mahalanobis

### Execution

1. **Ensure Blue Team models exist**
```bash
ls ../data/*.pkl ../data/*detector*.pt
```

2. **Open Red Team notebook**
```bash
jupyter notebook red-team.ipynb
```

3. **Run all cells sequentially**
- Generates adversarial examples
- Tests evasion against each detector
- Produces visualizations and metrics

**Runtime:** ~1.5 hours on Tesla T4 GPU

**Environment:** Tested on Google Colab with T4 GPU

## 📁 Output Files

**Adversarial Examples** (saved to `../data/`):
- `test_subset_clean.pt` - Clean samples (every 10th image from dataset)
- `adversarial_attack_results.pt` - PGD-generated adversarial examples

**Analysis Results:**
- Attack success rate metrics
- Evasion rate analysis per detector
- Visual comparison plots
- Perturbation distribution histograms

## Attack Implementation

**Gradient Computation:**
- Backward pass through ResNet-8 to compute ∂L/∂x
- L2 gradient normalization for consistent step sizes
- Projection onto L2 ball for perturbation bounding

**Optimization:**
- Untargeted: Maximize cross-entropy loss for true class
- Targeted: Minimize cross-entropy loss toward Class 5
- Adam-style iterative refinement

### Test Dataset Construction

- Systematic sampling: Every 10th image from CIFAR-10 (1,000 samples)
- Preserves class distribution
- Baseline accuracy: 92.1% (comparable to full dataset)

## 💡 Key Security Insights

### Defense Weaknesses Identified

1. **VAE Robustness:** Reconstruction-based detection proves most reliable against adversarial evasion
2. **OCSVM Instability:** Feature-space detection becomes unstable at extreme perturbations
3. **Mahalanobis Failure:** Critical inverse robustness - stronger attacks evade detection MORE easily
4. **Ensemble Necessity:** Single-layer defense insufficient; multi-detector systems recommended

### Recommended Countermeasures

**Immediate:**
- Prioritize VAE for adversarial filtering
- Implement ensemble OOD detection (Mahalanobis + alternatives)
- Add input preprocessing to disrupt gradient flow

**Long-term:**
- Adversarial training with PGD-generated samples
- Certified defense mechanisms (randomized smoothing)
- Continuous red team testing as upload patterns evolve

## 📄 Technical Report

Detailed security analysis available in: `red-team-report.pdf`
