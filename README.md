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