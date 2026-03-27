# Data Augmentation for FSHD Gene Expression Using Wavelet-GAN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green) ![PyWavelets](https://img.shields.io/badge/PyWavelets-latest-purple) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project addresses the **critical challenge of small sample sizes** in biomedical genomics by using **Generative AI to synthesize new gene expression samples**. Applied to the high-dimensional **FSHD microarray dataset** (50 samples, 33,298 gene features), the approach combines **Wavelet Transform decomposition** with a **Generative Adversarial Network (GAN)** to produce realistic synthetic samples — effectively doubling the training set while preserving biological signal structure.

Classifiers are benchmarked **before and after augmentation** to quantify the improvement from synthetic data.

---

## The Core Problem

> Biomedical datasets are often severely limited in sample count due to the rarity of diseases, cost of collection, and ethical constraints. With only **40 training samples** for 33,298 features, most classifiers face extreme overfitting risk.

This project proposes a novel two-stage solution:

```
Raw Gene Expression Data
        │
        ▼
Wavelet Transform Decomposition    ← Separate signal into frequency subbands
        │  cA (approximation) + cD (detail)
        ▼
GAN trained on cA subband          ← Learn the distribution of real samples
        │
        ▼
Synthetic cA coefficients          ← Generate 50 new samples
        │
        ▼
Wavelet Reconstruction             ← Rebuild full gene expression vectors
        │
        ▼
Augmented Dataset (40 real + 50 synthetic = 90 samples, balanced 45:45)
        │
        ▼
Classifier Training & Evaluation
```

---

## Dataset

- **Source:** GEO Dataset `GSE36398` (FSHD Series Matrix)
- **Samples:** 50 total (FSHD patients + Normal controls)
- **Features:** 33,298 gene probes
- **Labels:** Binary — `FSHD` vs `Normal`
- **Train/Test Split:** 80/20 stratified → 40 train, 10 test

---

## Methodology

### Stage 1 — Baseline Classification (Raw Features)
Six classifiers trained directly on scaled 33,298-dimensional features to establish a pre-augmentation baseline:

| Classifier |
|---|
| Random Forest |
| SVM (RBF kernel) |
| k-Nearest Neighbors |
| Naive Bayes |
| Logistic Regression |
| Gradient Boosting |

Metrics: **Accuracy, F1 Score, Recall, Confusion Matrix, ROC-AUC**

---

### Stage 2 — Wavelet Transform Decomposition

The training data is decomposed using **Daubechies-1 (db1) wavelet** at level 1:

```python
cA, cD = wt_decompose(X_train_scaled, wavelet='db1', level=1)
# cA → approximation coefficients (low-frequency, dominant signal)
# cD → detail coefficients (high-frequency, fine structure)
```

This separates the gene expression signal into two complementary subbands, allowing the GAN to model only the dominant signal structure (cA).

---

### Stage 3 — GAN Architecture

A lightweight GAN is trained on the **cA approximation subband**:

**Generator:**
```
Noise (z=10) → Dense(128, ReLU) → Dense(dim, Linear) → Synthetic cA
```

**Discriminator:**
```
cA input → Dense(128, ReLU) → Dense(1, Sigmoid) → Real / Fake
```

| GAN Hyperparameter | Value |
|---|---|
| Latent Noise Dimension | 10 |
| Hidden Units | 128 |
| Batch Size | 8 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |

---

### Stage 4 — Synthetic Sample Generation & Reconstruction

```python
# Generate 50 synthetic samples
cA_synth = generator.predict(noise)        # GAN-synthesized approximation coeffs
cD_synth = cD[random_indices]              # Detail coeffs sampled from real data

X_synth = wt_reconstruct(cA_synth, cD_synth)   # Reconstruct full gene vectors
```

**Augmented dataset composition:**

| Split | Samples | FSHD | Normal |
|---|---|---|---|
| Original training | 40 | ~20 | ~20 |
| Synthetic | 50 | 25 | 25 |
| **Augmented total** | **90** | **45** | **45** |

The augmentation also achieves **perfect class balance** (45 FSHD : 45 Normal).

---

### Stage 5 — Post-Augmentation Classification

All six classifiers are retrained on the augmented dataset and evaluated on the same held-out test set for direct comparison with the baseline.

---

## Project Structure

```
├── data-augmentation-using-generative-ai.ipynb   # Main notebook
└── README.md
```

---

## Requirements

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib PyWavelets
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/wavelet-gan-augmentation.git
   cd wavelet-gan-augmentation
   ```

2. **Download the dataset** from [GEO GSE36398](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36398) and place it at:
   ```
   /kaggle/input/fshd-dataset/Dataset2_GSE36398_series_matrix (1).txt
   ```
   *(Update `file_path` in Cell 0 for local execution)*

3. **Run the notebook**
   ```bash
   jupyter notebook data-augmentation-using-generative-ai.ipynb
   ```

---

## Key Concepts

- **Wavelet Transform:** Decomposes signals into frequency subbands. The approximation coefficients (cA) capture the dominant low-frequency structure — ideal for generative modeling.
- **GAN (Generative Adversarial Network):** A generator and discriminator trained in competition. The generator learns to produce samples indistinguishable from real data.
- **Wavelet-domain GAN:** By generating in wavelet space rather than raw feature space, the model better preserves the underlying signal structure of gene expression data.
- **Data Augmentation in Genomics:** Synthetic sample generation is especially valuable in rare disease research where collecting more real samples is infeasible.
- **FSHD:** A genetic muscular dystrophy with distinct gene expression signatures detectable via microarray analysis.

---

## Technologies Used

- **TensorFlow / Keras** — GAN architecture (Generator, Discriminator)
- **PyWavelets (pywt)** — Wavelet decomposition and reconstruction
- **scikit-learn** — Classifiers, preprocessing, evaluation metrics
- **Pandas / NumPy** — Data manipulation
- **Matplotlib** — ROC curve visualization

---

## Related Projects

This notebook is part of a series studying FSHD gene expression classification:

- [VAE-Based Dimensionality Reduction for FSHD Classification](../vae-dimensionality-reduction/)

---

## Citation

If you use this work, please cite the dataset:

> GEO Accession: [GSE36398](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36398)

---

## License

This project is licensed under the MIT License.
