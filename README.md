# Face Mood Recognition

## 1. Title & Team

**Project Title:** Face Mood Recognition

**Team Members:**
- Abzalbek Ibrokhimov — 220587  
- Islombek Pulatov — 220358  
- Azizbek Savkimov — 220879

**GitHub Repository:** [https://github.com/asavkimov/face-mood-recognition](https://github.com/asavkimov/face-mood-recognition)

---

## 2. Abstract

Facial expressions are one of the most natural ways humans convey emotions.  
We aim to develop a **Face Mood Recognition system** that automatically classifies facial expressions into categories such as **happiness, sadness, anger, fear, surprise, disgust, and neutrality**.

We will use **CNN-based models** like *MobileNetV2* fine-tuned on **FER2013** or **AffectNet**, targeting **70% accuracy** and **real-time inference**.  
Expected outcomes include a trained model, GitHub code, and a video demo.

---

## 3. Problem & Motivation

Understanding human emotions is critical for **mental health monitoring**, **adaptive UIs**, and **robotics**.  
Traditional vision systems fail under real-world variability; deep learning overcomes this.

**Goal:** Emotion recognition accuracy ≥70%, latency ≤50ms.  
**Beneficiaries:** Educators, developers, and researchers.

---

## 4. Related Work

Key works include:
- Goodfellow et al. (2013)
- Mollahosseini et al. (2017)
- Li & Deng (2020)
- Corneanu et al. (2016)
- Zhang et al. (2019)

Our approach balances accuracy and efficiency using **transfer learning** with *MobileNetV2* / *EfficientNet-B0*.

---

## 5. Data & Resources

**Datasets:**
- FER2013
- AffectNet
- CK+

**Compute:** Google Colab GPU (T4 / A100)  
**Frameworks:** PyTorch, OpenCV  

**Ethics:** Public datasets only, anonymized results.

---

## 6. Method

**Baseline:** CNN trained on FER2013.  
**Proposed:** Transfer learning with *MobileNetV2*:
- Preprocessing with MTCNN
- Normalization and augmentation
- Fine-tuning upper layers
- Weighted loss function

**Ablation Studies:**
- Effect of normalization  
- Transfer learning vs. training from scratch  
- MobileNet vs. ResNet comparison

---

## 7. Experiments & Metrics

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Latency
- Model size

**Targets:**  
Accuracy ≥70%, Latency ≤50ms

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|---------|-------------|
| Limited dataset diversity | Medium | Combine FER2013 + AffectNet |
| GPU limits | Medium | Use smaller batch sizes, cloud credits |
| Overfitting | High | Apply regularization, early stopping |
| Domain shift | Medium | Use augmentation, face alignment |

---

## 9. Timeline & Roles

**6-week roadmap:**
| Week | Task | Responsible |
|------|-------|-------------|
| 1 | Data preparation | Abzalbek |
| 2 | Baseline model training | Abzalbek |
| 3 | Transfer learning setup | Islombek |
| 4 | Hyperparameter tuning | Aziz |
| 5 | Evaluation & ablation | Team |
| 6 | Report + demo | Team |

**Roles:**  
- **Azizbek:** Coordination, tuning  
- **Abzalbek:** Baseline development  
- **Islombek:** Transfer learning, documentation

---

## 10. Expected Outcomes

**Deliverables:**
- Code repository  
- Final report  
- Video demo  
- Poster  

**Stretch goal:** Deployable model (TensorFlow Lite for mobile devices)

---

## 11. Ethics & Compliance

All datasets are **academic/public**.  
No private data collected.  
Bias analysis will be conducted across gender and ethnicity.  
No IRB approval required.

---

## 12. References

1. Goodfellow et al. (2013). *FER2013 Dataset.*  
2. Mollahosseini et al. (2017). *AffectNet.*  
3. Li & Deng (2020). *Deep Emotion Recognition.*  
4. Corneanu et al. (2016). *Survey on Facial Expression Recognition.*  
5. Zhang et al. (2019). *Efficient Emotion Recognition Using MobileNet.*  
6. Cohn & Kanade (2000). *CK+ Dataset.*

---

## 13. Baseline Model: Design & How to Run

We provide two baseline image classifiers for FER2013 (7 classes):

- `ResNet18Baseline` (transfer learning, ImageNet weights; recommended)
- `SmallCNN` (lightweight custom CNN)

### Data expected
Use the already prepared splits under `data/processed/fer2013/{train,val,test}.csv` created by our EDA scripts.

### Install deps
```
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Train (quick smoke test)
Run a 1-epoch sanity check on a small subset (10 images per class) — finishes in a few minutes on CPU:
```
python3 scripts/train_image_baseline.py \
  --model resnet18 \
  --epochs 1 \
  --freeze_epochs 0 \
  --batch_size 64 \
  --img_size 224 \
  --limit_per_class 10 \
  --no_amp
```

### Train (full)
Fine-tune ResNet18 end-to-end for ~15 epochs:
```
python3 scripts/train_image_baseline.py \
  --model resnet18 \
  --epochs 15 \
  --freeze_epochs 2 \
  --batch_size 128 \
  --img_size 224
```
Use `--model smallcnn` for the lightweight baseline.

### Outputs
- Checkpoints: `outputs/models/`
- Metrics/logs: `outputs/metrics/train_log.csv`, `outputs/metrics/*classification_report*.json`
- Plots: `outputs/plots/confusion_matrix_{val,test}.png`
- Summary: `outputs/metrics/last_train_summary.json` (also appended to README automatically after a run)

### Notes
- Images are grayscale but models expect 3 channels; we repeat the channel to convert to RGB.
- Loss uses class weights computed from training distribution.
- Mixed precision is enabled by default when CUDA is available.
- Set a different seed with `--seed` for reproducibility.

### Baseline performance (expected ranges)
Numbers vary by seed and augmentation; typical ranges reported on FER2013:
- ResNet18 (transfer learning, 224px): 68–73% test accuracy
- SmallCNN (from scratch): 58–65% test accuracy

Run the training to generate your project’s exact metrics and plots.
