# Speech Assignment 2 - M23CSA017

**Name**: Prabhat Ranjan



## Task 1B: Speech Enhancement and Speaker Verification in Multi-Speaker Scenarios

###  Objective

To verify speakers from mixed audio recordings by:

- Using a **pre-trained wav2vec2 XLSR model**
- Applying **LoRA** and **ArcFace** for fine-tuning
- Using **SepFormer** for speech separation

###  Dataset

- **VoxCeleb1** – For evaluation (1,251 speakers, 150k utterances)
- **VoxCeleb2** – For fine-tuning and multi-speaker training (6,112 speakers)

###  Methodology

- Fine-tuned wav2vec2 with **LoRA (r=8, alpha=8, dropout=0.05)** and **ArcFace loss**
- SepFormer used for speech separation
- Evaluation pre- and post-separation on speaker verification metrics

###  Results

| Metric                  | Pre-trained | Fine-tuned |
| ----------------------- | ----------- | ---------- |
| Equal Error Rate (EER)  | 47.93%      | 41.45%     |
| TAR @ 1% FAR            | 2.46%       | 7.99%      |
| Identification Accuracy | 34.56%      | 63.70%     |

####  Separation Quality (SepFormer)

| Metric | Mean     | Std. Dev |
| ------ | -------- | -------- |
| SDR    | 3.73 dB  | 7.32 dB  |
| SIR    | 19.38 dB | 11.34 dB |
| SAR    | 5.21 dB  | 4.45 dB  |
| PESQ   | 1.22     | 0.25     |

####  Post-Separation Verification Accuracy

| Model       | Rank-1 Accuracy |
| ----------- | --------------- |
| Fine-tuned  | 53.50%          |
| Pre-trained | 44.50%          |

---

##  Task 2: MFCC-Based Comparative Analysis of Indian Languages

###  Task A: MFCC Feature Extraction and Visualization

- Dataset: **Audio Dataset with 10 Indian Languages** from Kaggle
- Selected Languages: **Hindi, Punjabi, Telugu**
- Tools: **Librosa**, **Matplotlib**

####  Key Observations:

- Hindi: Denser, more varied MFCCs
- Punjabi: Strong energy in 0–3 kHz
- Telugu: Clear formant transitions

### Statistical Analysis

- **MFCC Mean & Variance** captured core spectral properties
- First few coefficients contributed most to signal energy
- **Variance Analysis**: Hindi showed the highest variability

---

###  Task B: Language Classification using MFCCs

####  Feature Matrix

- Shape: `(10000, 26)`
- 13 MFCC means + 13 MFCC variances per sample

####  Model

- Classifier: **Random Forest**
- Accuracy: **84.80%**

#### Performance Summary (F1 Score)

- High: Hindi (0.96), Tamil (0.97), Telugu (0.97)
- Low: Punjabi (0.47), Gujarati (0.50)

---

## Challenges & Limitations

- **Speaker Variability**: Affected MFCC robustness
- **Background Noise**: Reduced classification accuracy
- **Accent and Dialectal Variation**: Caused misclassifications
- **Low-resource Languages**: Limited training data
- **Domain Mismatch**: Clean training vs. distorted test audio
- **Noisy Separation**: Artifacts introduced post-Separation

---

## 🛠️ Tools & Libraries Used

- `torch`, `torchaudio`
- `transformers`
- `librosa`, `matplotlib`, `seaborn`
- `mir_eval`, `pesq`
- `sklearn`, `scipy`

---
