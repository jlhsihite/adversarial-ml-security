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

## 🛠️ Dependencies

- **Python 3.10+**
- **PyTorch 2.8** - Deep learning framework, gradient computation
- **NumPy 2.2** - Numerical operations
- **scikit-learn 1.6** - OCSVM, evaluation metrics
- **Matplotlib, Seaborn** - Visualization
- **joblib** - Model serialization

---
## 📊 Dataset

**Dataset:** CIFAR-10 grayscale (32×32 images, 10 classes)

### Data Splits

- **Anomaly Detection:** 
  - Validation: 2,000 labeled samples (1,000 normal + 1,000 corrupted)
  - Test: 10,000 unlabeled samples
  
- **Out-of-Distribution Detection:**
  - Validation: 5,000 labeled samples (2,500 in-distribution + 2,500 OOD)
  - Test: 10,000 unlabeled samples
  
- **Red Team:** 1,000 clean samples for attack generation


**Note:** Datasets are not included in repository.
See data/README.md for instructions for running with your own data.

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
├── .gitignore                      
├── README.md                       
├── requirements.txt                
├── blue-team/
│   ├── README.md                   
│   ├── blue-team.ipynb             
│   └── blue-team-report.pdf        
├── data/
│   └── README.md                   
└── red-team/
    ├── README.md                   
    ├── red-team.ipynb              
    └── red-team-report.pdf         
```
---

## 🔗 Related Work

This project builds on foundational research in:
- Adversarial robustness (Goodfellow et al., 2014)
- Out-of-distribution detection (Lee et al., 2018 - Mahalanobis)
- Anomaly detection in deep learning (An & Cho, 2015 - VAE)
