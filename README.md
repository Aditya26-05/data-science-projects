# Data Science & Machine Learning Projects

[![Author](https://img.shields.io/badge/Author-Aditya26--05-blue.svg)](https://github.com/Aditya26-05)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A curated repository of empirical machine learning research, applied data science workflows, and statistical modeling implementations. Each project within this repository is organized as a self-contained module with its own source code, analysis notebooks, benchmarking scripts, and documentation.

---

## 📚 Projects Portfolio

| # | Project Directory | Domain / Focus | Key Technologies | Status |
| :-: | :--- | :--- | :--- | :-: |
| 1 | [**Label Noise Robustness Study**](label-noise-robustness-study/) | Empirical ML Research, Label Noise, Robustness, Data Cleaning | `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `seaborn` | ✅ Completed |

---

## 🔬 Featured Project: Label Noise Robustness & Mitigation Study

### [📂 label-noise-robustness-study/](label-noise-robustness-study/)

An empirical investigation into how synthetic symmetric label noise degrades the generalization performance of canonical machine learning paradigms (**Logistic Regression**, **Random Forest**, and **XGBoost**), paired with robust mitigation strategies:

- **Theoretical Inductive Bias Hypotheses**: Formalizes why boosting suffers from residual chasing on corrupted labels, while bagging buffers errors via consensus voting up to $\approx 30\%$ noise.
- **Empirical Degradation Benchmarks**: Tested on Wisconsin Breast Cancer across noise levels $\eta \in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]$ over multiple random seeds. Demonstrated XGBoost drops by $-27.9\%$ at 40% noise, whereas Random Forest maintains robustness up to 30%.
- **Mitigation & Recovery**:
  - **Out-of-Fold (OOF) Confident Filtering**: 5-fold cross-validated cleaner achieving **94.7% noise detection precision**.
  - **Noise-Aware Soft Sample Reweighting**: Continuous loss attenuation recovering Random Forest accuracy to **90.91%** (+6.29% gain over noisy baseline).
  - **Inductive Regularization**: Tuned hyperparameter constraints buffering gradient swings.
- **2D Boundary Visualizations**: Generates geometric decision frontier contours under clean, noisy, and mitigated regimes on non-linear synthetic data.

👉 **[Read Full Project Documentation & Results](label-noise-robustness-study/README.md)**

---

## 📁 Repository Structure

```text
data-science-projects/
├── README.md                              # Root repository index and portfolio guide
├── .gitignore                             # Git ignore rules for Python, notebooks, and OS artifacts
│
└── label-noise-robustness-study/          # Project 1: Label Noise Robustness
    ├── README.md                          # Detailed project report & theoretical background
    ├── requirements.txt                   # Project dependencies
    ├── main.py                            # End-to-end experiment pipeline
    ├── notebooks/
    │   └── label_noise_study.ipynb        # Interactive Jupyter walkthrough
    ├── src/
    │   ├── experiment.py                  # Multi-seed benchmarking harness
    │   ├── mitigation.py                  # OOF cleaner & soft sample reweighting
    │   ├── models.py                      # Model configurations (LR, RF, XGBoost)
    │   ├── noise_injector.py              # Synthetic label noise generation
    │   └── visualization.py               # Publication-grade plotting routines
    └── results/
        ├── figures/                       # Generated degradation curves & boundaries
        ├── mitigation_results.csv
        └── noise_degradation_metrics.csv
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Aditya26-05/data-science-projects.git
cd data-science-projects
```

### 2. Run a Specific Project
Navigate to the desired project directory and install its requirements:
```bash
cd label-noise-robustness-study
pip install -r requirements.txt
python main.py
```

---

## 👤 Author

**Aditya Raghuvanshi**
- GitHub: [@Aditya26-05](https://github.com/Aditya26-05)
