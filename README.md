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
We aim to develop a **Face Mood Recognition system** that automatically classifies facial expressions into 7 emotion categories: **Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised**.

We use a **custom CNN model** trained on **FER2013** dataset, targeting **70% accuracy** and **real-time inference**.  
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

Our approach uses a **custom CNN architecture** trained from scratch on FER2013 for emotion recognition.

---

## 5. Data & Resources

**Datasets:**

- FER2013

**Compute:** Google Colab GPU (T4 / A100)  
**Frameworks:** TensorFlow/Keras, OpenCV

**Ethics:** Public datasets only, anonymized results.

---

## 6. Method

**Baseline:** Custom CNN trained on FER2013 from scratch.

**Architecture:**

- 4 convolutional layers (32, 64, 128, 128 filters)
- MaxPooling and Dropout layers for regularization
- Dense layers (1024 units) with softmax output
- Input: 48×48 grayscale images
- Output: 7 emotion classes

**Preprocessing:**

- Face detection using Haar Cascade
- Image normalization (rescale to [0, 1])
- Data augmentation via ImageDataGenerator

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

| Risk                      | Impact | Mitigation                                   |
| ------------------------- | ------ | -------------------------------------------- |
| Limited dataset diversity | Medium | Use data augmentation techniques             |
| GPU limits                | Medium | Use smaller batch sizes, cloud credits       |
| Overfitting               | High   | Apply dropout regularization, early stopping |
| Domain shift              | Medium | Use augmentation, face detection             |

---

## 9. Timeline & Roles

**6-week roadmap:**
| Week | Task | Responsible |
|------|-------|-------------|
| 1 | Data preparation | Abzalbek |
| 2 | Baseline model training | Abzalbek |
| 3 | Model optimization | Islombek |
| 4 | Hyperparameter tuning | Aziz |
| 5 | Evaluation & testing | Team |
| 6 | Report + demo | Team |

**Roles:**

- **Azizbek:** Coordination, tuning
- **Abzalbek:** Baseline development
- **Islombek:** Model optimization, documentation

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

1. Goodfellow et al. (2013). _FER2013 Dataset._
2. Mollahosseini et al. (2017). _AffectNet._
3. Li & Deng (2020). _Deep Emotion Recognition._
4. Corneanu et al. (2016). _Survey on Facial Expression Recognition._
5. Zhang et al. (2019). _Efficient Emotion Recognition Using MobileNet._
6. Cohn & Kanade (2000). _CK+ Dataset._

---

## 13. Baseline Model: Design & How to Run

We provide a custom CNN baseline model for FER2013 emotion recognition (7 classes):

- **Custom CNN** — A sequential convolutional neural network with:
  - 4 convolutional layers (32, 64, 128, 128 filters)
  - MaxPooling and Dropout layers for regularization
  - Dense layers (1024 units) with final softmax output for 7 emotion classes
  - Input size: 48×48 grayscale images
  - Output: 7 emotion classes (angry, disgusted, fearful, happy, neutral, sad, surprised)

### Data expected

The data is organized as image directories under `src/data/` with the following structure:

- `src/data/train/` — training images organized by emotion class (angry, disgusted, fearful, happy, sad, surprised, neutral)
- `src/data/test/` — test images organized by emotion class (same structure)

The data is prepared by running `src/dataset_prepare.py`, which reads from `src/fer2013.csv` and creates the directory structure with images organized by emotion class.

### Install deps

```
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

source .venv/bin/activate
python src/dataset_prepare.py

python src/emotions.py --mode train
python src/emotions.py --mode display
```

### Train

Train the custom CNN model on FER2013:

```
python src/emotions.py --mode train
```

The model will train for 50 epochs with:

- Batch size: 64
- Input size: 48×48 grayscale images
- Optimizer: Adam (learning rate: 0.0001)
- Loss: Categorical crossentropy

### Outputs

- Model: `src/model.h5`
- Plots: `src/plot.png`
- Data (test, train): `src/data/test` `src/data/train`

### Notes

- Images are grayscale (48×48 pixels) and processed as single-channel.
- The model uses dropout layers (0.25 and 0.5) for regularization.
- Training uses ImageDataGenerator for data loading and preprocessing.
- Model weights are saved to `src/model.h5` after training.

### Display mode

Run real-time emotion detection from webcam:

```
python src/emotions.py --mode display
```

The model will use Haar Cascade for face detection and display emotions in real-time. Press 'q' to quit.
