# Adversarial ML Security: Blue Team Defense vs Red Team Attack
**Author:** Jessica Sihite  
A comprehensive adversarial machine learning security assessment implementing multi-layered detection systems and evaluating their robustness against gradient-based attacks.

**Grade:** H1 (20/25)

## 🎯 Project Overview

This project simulates a real-world ML security scenario where a company's image classification system accepts user-uploaded data for continuous model improvement. The system faces two critical threats:

1. **Data Integrity Attacks:** Corrupted images that degrade model performance
2. **Adversarial Attacks:** Maliciously crafted inputs designed to evade detection

The project implements a **Blue Team vs Red Team** methodology:
- **Blue Team:** Builds multi-layered defense systems to filter malicious inputs (anomaly and out-of-distribution)
- **Red Team:** Develops adversarial attacks to identify defense vulnerabilities

---
## 🛡️ Blue Team: Defense Architecture

#### 1. Shallow Anomaly Detection: One-Class SVM (OCSVM)
- **Approach:** Learns decision boundary around normal data in 512D feature space
- **Performance:** 92.2% accuracy, 96.9% AUROC
- **Strength:** Threshold-stable across dynamic data distributions
- **Trade-off:** Misses 53 pixel-level corruptions due to feature compression

#### 2. Deep Anomaly Detection: Variational Autoencoder (VAE)
- **Approach:** Reconstruction-based detection in 1024D pixel space
- **Performance:** 94.2% accuracy, 96.3% AUROC
- **Strength:** Detected 37 more anomalies than OCSVM via pixel-level analysis
- **Trade-off:** Lower ranking consistency, requires threshold recalibration

#### 3. Out-of-Distribution Detection: Mahalanobis Distance
- **Approach:** Statistical distance in learned feature space
- **Performance:** 95.1% accuracy, 98.2% AUROC
- **Key Achievement:** **8.2× lower false positive rate** than Maximum Softmax Probability
- **Strength:** Robust to confidence miscalibration, leverages intermediate representations

### Defense Results Summary

| Detector | Accuracy | AUROC | AUPRC | Key Strength |
|----------|----------|-------|-------|--------------|
| **OCSVM** | 92.2% | 96.9% | 94.5% | Threshold stability |
| **VAE** | 94.2% | 96.3% | 92.9% | Pixel-level detection |
| **Mahalanobis** | 95.1% | 98.2% | 98.7% | Low false positives (8.2× improvement) |

---
## ⚔️ Red Team: Adversarial Attack Assessment

### Attack Methodology: Projected Gradient Descent (PGD)

**Implementation:**
- L2-norm constrained perturbations for imperceptibility
- Untargeted attacks: Force any misclassification
- Targeted attacks: Force specific misclassification to Class 5
- Iterative optimization with gradient normalization

**Configuration:**
- Step sizes: 10⁻⁵ to 10¹ (logarithmic spacing)
- Epsilon: 12.0 (optimized for detection evasion vs attack success)
- Maximum iterations: 100
- Early stopping: Check every 20 iterations

### Attack Performance

| Step Size (α) | Untargeted ASR | Targeted ASR | Mean L2 Norm | Accuracy Drop |
|---------------|----------------|--------------|--------------|---------------|
| 10⁻⁵ | 1.7% | 0.1% | 22.350 | 1.6% |
| 10⁻³ | 21.4% | 0.7% | 22.431 | 19.8% |
| 10⁻¹ | 88.0% | 28.0% | 22.547 | 81.2% |
| 10⁰ | 97.1% | 46.9% | 22.582 | 89.5% |
| 10¹ | **99.4%** | 56.6% | 23.202 | **91.3%** |

**Key Findings:**
- Attack success depends on optimization trajectory, not just perturbation magnitude
- Targeted attacks 42% harder than untargeted (stricter constraints)
- Visual imperceptibility maintained even at extreme step sizes

---

## 🔓 Defense Evasion Analysis

### Critical Vulnerability Discovered

Red Team testing revealed **differential robustness** across detection layers:

| Detector | Evasion Rate (α=10⁻¹) | Evasion Rate (α=10¹) | Robustness Assessment |
|----------|----------------------|---------------------|----------------------|
| **VAE** | 5.6% | **3.0%** | ✅ Most Robust - Consistent across all attack strengths |
| **OCSVM** | 26.5% | 8.2% (unstable) | ⚠️ Feature-space instability at extremes |
| **Mahalanobis** | 28.4% | **46.4%** | ❌ Inverse robustness - Fails at high perturbations |

### Exploitation Scenario

**Critical Finding:** Adversarial samples can **exploit weak OOD detection** to:
1. Bypass Mahalanobis distributional filtering (46.4% evasion rate)
2. Evade OCSVM anomaly detection (48.3% evasion at α=10⁰)
3. Contaminate training pipeline with mislabeled examples
4. Progressively degrade model performance over retraining cycles

**Real-world Impact:** Adversaries can inject visually-normal product images that cause systematic misclassification, damaging client trust and business operations.

---

## 🛠️ Dependencies

- **Python 3.10+**
- **PyTorch 2.8** - Deep learning framework, gradient computation
- **NumPy 2.2** - Numerical operations
- **scikit-learn 1.6** - OCSVM, evaluation metrics
- **Matplotlib, Seaborn** - Visualization
- **joblib** - Model serialization

---

## 📊 Dataset

- **Base:** CIFAR-10 grayscale (32×32 images, 10 classes)
- **Anomaly Detection:** 2,000 validation + 10,000 test samples (corrupted images)
- **OOD Detection:** 5,000 validation + 10,000 test samples (out-of-distribution content)
- **Red Team:** 1,000 clean samples for attack generation

**Note:** Datasets are not included in repository.

---

## 🚀 Execution

### 1. Clone Repository
```bash
git clone https://github.com/jlhsihite/adversarial-ml-security.git
cd adversarial-ml-security
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Blue Team (Defense Training)
```bash
jupyter notebook blue-team/blue-team.ipynb
# Runtime: ~15-20 minutes on GPU
```

### 4. Run Red Team (Attack Testing)
```bash
jupyter notebook red-team/red-team.ipynb
# Runtime: ~1.5 hours on Tesla T4 GPU
# Requires Blue Team models to be trained first
```

---

## 📁 Repository Structure
```
adversarial-ml-security/
├── README.md                       
├── blue-team/
│   ├── README.md                   
│   ├── blue-team.ipynb             
│   └── blue-team-report.pdf        
├── red-team/
│   ├── README.md                   
│   ├── red-team.ipynb              
│   └── red-team-report.pdf         
├── requirements.txt                
```
---

## 💡 Key Insights & Recommendations

### Defense Architecture Trade-offs

1. **OCSVM:** Best for production stability - consistent performance across dynamic upload patterns
2. **VAE:** Best for comprehensive protection - detects 37 additional corruptions via pixel-level analysis
3. **Mahalanobis:** Best for false positive reduction - 8.2× improvement over baseline methods

### Security Recommendations

**Immediate Actions:**
- Deploy VAE as primary anomaly filter (highest robustness to adversarial evasion)
- Implement ensemble OOD detection beyond Mahalanobis alone
- Add continuous threshold recalibration as data distributions evolve

**Long-term Hardening:**
- Adversarial training with PGD-generated samples
- Input transformations to disrupt gradient-based attacks
- Multi-scale feature analysis combining pixel and semantic spaces

---

## 🔗 Related Work

This project builds on foundational research in:
- Adversarial robustness (Goodfellow et al., 2014)
- Out-of-distribution detection (Lee et al., 2018 - Mahalanobis)
- Anomaly detection in deep learning (An & Cho, 2015 - VAE)
