# 🩺 AI-Assisted Breast Ultrasound Analysis

## EfficientNet-B3-Based Breast Lesion Classification with Grad-CAM++ Explainability

**Medical Imaging × Artificial Intelligence × Explainable AI**

An independent medical imaging AI project exploring whether deep learning can classify breast ultrasound lesions as **Benign** or **Malignant**, while also investigating **where the model is looking when making its prediction**.

---

## 🚀 Try the Live App | 📊 View Results | 🔬 Explainability Analysis

🌐 **[Live Streamlit Application](https://ai-assisted-breast-ultrasound-3mupqtp6ehvq6jrkwsjst8.streamlit.app/)** · 
🤗 **[Trained Model](https://huggingface.co/zara14ashraf/busbra-dual-effnet-b3-final)** · 
💻 **[Source Code](https://github.com/zara14ashraf-web/AI-Assisted-Breast-Ultrasound)**
> **Note:** The deployed model is intended for research and educational purposes only. It is not a clinical diagnostic system and must not be used independently for diagnosis or treatment decisions.

---

# 🖥️ Application Preview

The deployed Streamlit application allows users to upload a breast ultrasound image and explore the model's analysis through:

* AI prediction: **Benign or Malignant**
* Benign and Malignant probability outputs
* Model separation
* Probability distribution
* Grad-CAM++ visual explanation
* Representative BUS-BRA sample cases
* Reference lesion-mask comparison
* Downloadable AI analysis report

The application was designed to make the project understandable not only to people familiar with AI, but also to users who may be seeing concepts such as Grad-CAM++ for the first time.

---

# Introduction

Breast ultrasound is widely used for the evaluation of breast lesions and can provide important information about lesion characteristics. However, image interpretation remains dependent on professional expertise and the overall clinical context.

This project explores a focused research question:

> **Can a deep-learning model distinguish between benign and malignant breast ultrasound lesions, and can we investigate whether its attention overlaps with the documented lesion region?**

The first part of the question is about **classification**.

The second is about **explainability**.

A model may achieve a good classification result, but accuracy alone does not reveal what image information influenced that prediction. For this reason, the project extends beyond simple classification by using **Grad-CAM++** and quantitatively comparing model attention with the available reference lesion masks.

The objective is not to replace radiologists or other healthcare professionals. Instead, this project explores how AI-assisted analysis and explainable AI can be investigated within medical imaging research.

---

# 🎯 Why This Project?

As a Medical Imaging Technologist, I wanted this project to explore more than simply:

> *"Can an AI model achieve a high accuracy?"*

The more interesting question was:

> **"What is the model actually looking at when it makes that prediction?"**

This led to a project that combines:

* Breast ultrasound image analysis
* Deep-learning classification
* Model comparison
* Threshold selection
* Image-level evaluation
* Case-level evaluation
* Confidence and calibration analysis
* Grad-CAM++ explainability
* Quantitative lesion-attention comparison
* Interactive deployment

The final project therefore focuses on both **what the model predicts** and **how its visual attention relates to the lesion**.

---

# 🔍 What Does the System Do?

The deployed model performs **binary breast lesion classification**.

| Class            | Description                                      |
| ---------------- | ------------------------------------------------ |
| 🟢 **Benign**    | Image classified into the benign lesion class    |
| 🔴 **Malignant** | Image classified into the malignant lesion class |

The application does **not** perform:

* Normal vs Benign vs Malignant classification
* Automated diagnosis of every breast abnormality
* Lesion segmentation
* Independent clinical decision-making

### Workflow

```text
Breast Ultrasound Image
          ↓
     Preprocessing
          ↓
    EfficientNet-B3
          ↓
 Benign / Malignant Outputs
          ↓
     Decision Threshold
          ↓
      AI Prediction
          ↓
 Probability + Model Separation
          ↓
   Grad-CAM++ Explanation
```

The model's final output is a **binary prediction**, while Grad-CAM++ provides a visual representation of image regions associated with that prediction.

---

# 📊 Dataset

The project uses the **BUS-BRA breast ultrasound dataset**.

Dataset verification confirmed:

* **1,875 ultrasound images**
* **1,875 corresponding lesion masks**
* **1,875 / 1,875 verified image-mask matches**

The availability of both ultrasound images and lesion masks made it possible to evaluate not only classification performance but also the relationship between model attention and the documented lesion region.

### Final Classification Task

The deployed model focuses specifically on:

> **Benign vs Malignant**

Class mapping:

| Label | Class     |
| ----- | --------- |
| `0`   | Benign    |
| `1`   | Malignant |

---

# 🧠 Methodology

The project followed an iterative development workflow rather than relying on a single model from the beginning.

```text
Dataset Verification
        ↓
Image Preparation
        ↓
Initial Baseline Models
        ↓
Class Balancing & Augmentation
        ↓
Lesion-Focused Approaches
        ↓
Mask-Guided Experiments
        ↓
Dual-View & Ensemble Strategies
        ↓
Alternative Classical / Nonlinear Approaches
        ↓
Final EfficientNet-B3 Training
        ↓
Validation-Based Threshold Selection
        ↓
Untouched Test Evaluation
        ↓
Confidence & Calibration Analysis
        ↓
Grad-CAM++ Evaluation
        ↓
Lesion-Mask Comparison
        ↓
Case-Level Evaluation
        ↓
Streamlit Deployment
```

The final model was therefore selected after exploring multiple strategies rather than assuming that a single architecture would automatically be optimal.

---

# 🔬 Model Development Journey

Several architectures and strategies were investigated during development.

| Model / Strategy                               |          Validation |       Test | Development Outcome                              |
| ---------------------------------------------- | ------------------: | ---------: | ------------------------------------------------ |
| EfficientNet-B0                                |              79.49% |     69.23% | Initial baseline                                 |
| EfficientNet-B0 + Augmentation + Class Weights |              79.49% |     74.36% | Improved test performance                        |
| ResNet-50                                      |              53.85% |          — | Underperformed                                   |
| Lesion-Focused EfficientNet                    |                   — | **82.05%** | Major improvement                                |
| Mask-Guided EfficientNet                       |         **92.31%*** | **82.05%** | Strong validation but weaker test generalization |
| Dual-View Fusion                               |              79.18% |          — | Explored, not selected                           |
| Weighted Dual-View Ensemble                    |          **79.73%** |          — | Explored, not selected                           |
| PCA + Logistic Regression                      |          **75.07%** |          — | Alternative baseline                             |
| Nonlinear Dual-View Classifier                 |              78.36% | **81.88%** | Promising, but below final model                 |
| 🏆 **Final EfficientNet-B3**                   | Validation-selected | **88.35%** | **Selected and deployed**                        |

* Validation performance should not be interpreted as final generalization performance. The difference between validation and test performance was one reason to continue model development and evaluate alternative approaches.

### What This Development Process Showed

The experiments produced several useful findings.

#### 1. A larger architecture did not automatically perform better

The ResNet-50 experiment substantially underperformed compared with other approaches.

#### 2. Lesion-focused approaches improved performance

Moving beyond simple full-image classification produced a substantial improvement, with lesion-focused and mask-guided approaches reaching **82.05% test accuracy**.

#### 3. High validation performance was not enough

The Mask-Guided EfficientNet reached **92.31% validation accuracy**, but its final test accuracy remained **82.05%**.

This highlighted an important principle:

> **A strong validation score alone is not sufficient evidence of generalization.**

#### 4. More complex multi-view strategies were not automatically superior

Dual-view fusion, weighted ensembles, PCA-based classification, and nonlinear approaches were explored, but none ultimately exceeded the performance of the final selected model.

#### 5. The final model was selected based on the complete development process

The final EfficientNet-B3 produced the strongest verified result among the experiments documented here and was therefore selected for final evaluation and deployment.

---

# 🏆 Final Model

The final deployed model is a **single-image EfficientNet-B3 classifier** implemented using PyTorch and Torchvision.

| Component          | Configuration                        |
| ------------------ | ------------------------------------ |
| Architecture       | EfficientNet-B3                      |
| Framework          | PyTorch                              |
| Implementation     | `torchvision.models.efficientnet_b3` |
| Input              | Single breast ultrasound image       |
| Input Size         | 300 × 300                            |
| Task               | Binary lesion classification         |
| Output Classes     | Benign / Malignant                   |
| Decision Threshold | **0.52**                             |
| Explainability     | Grad-CAM++                           |

### Classification Architecture

```text
EfficientNet-B3
      ↓
1536-dimensional feature representation
      ↓
Linear Classification Layer
      ↓
2 Outputs
      ↓
Benign / Malignant
```

The final classifier configuration is:

```python
nn.Linear(1536, 2)
```

---

# 📈 Final Image-Level Results

The final model was evaluated on an **untouched test set of 309 images**.

| Metric               |     Result |
| -------------------- | ---------: |
| 🏆 **Accuracy**      | **88.35%** |
| Precision            |     81.90% |
| Recall / Sensitivity |     83.50% |
| F1-score             |     82.69% |
| 🏆 **ROC-AUC**       | **0.9364** |
| Test Images          |        309 |
| Decision Threshold   |   **0.52** |

---

# 🎯 Beyond a Single Accuracy Number

Accuracy alone does not fully describe a model.

For this reason, the final evaluation also included:

* Precision
* Recall / Sensitivity
* F1-score
* ROC-AUC
* Confidence analysis
* Calibration analysis
* Image-level evaluation
* Case-level evaluation
* Grad-CAM++ evaluation
* Quantitative lesion-attention comparison

This provides a broader view of the model than reporting accuracy alone.

---

# 🔎 Confidence & Calibration Analysis

The complete test set was also evaluated for prediction confidence.

| Metric                          |     Result |
| ------------------------------- | ---------: |
| Mean Confidence                 | **93.28%** |
| Correct Prediction Confidence   | **94.68%** |
| Incorrect Prediction Confidence |     82.60% |
| Expected Calibration Error      | **0.0570** |
| Predictions ≥80% Confidence     | **88.35%** |
| Predictions ≥90% Confidence     | **79.61%** |

---

# 🧪 Case-Level Evaluation

Breast ultrasound examinations may contain more than one image from the same case.

To investigate performance beyond individual image classification, the test set was also evaluated at the **case level**.

The test data contained:

* **170 unique cases**
* **139 cases with two views**
* **31 cases with one view**

### Case-Level Performance

| Metric          |     Result |
| --------------- | ---------: |
| 🏆 **Accuracy** | **90.00%** |
| Precision       |     81.67% |
| Recall          |     89.09% |
| F1-score        |     85.22% |
| 🏆 **ROC-AUC**  | **0.9505** |

### Case-Level Confusion Matrix

```text
[[104, 11],
 [  6, 49]]
```

Case-level evaluation achieved a higher accuracy than image-level evaluation:

```text
Image-Level Accuracy  →  88.35%
Case-Level Accuracy   →  90.00%
```

This provides an additional perspective on model performance when multiple views belonging to the same case are considered together.

---

# 🔬 Explainability: Beyond Classification

A central part of this project was the following question:

> **The model made a prediction—but where was it looking?**

To investigate this, the project uses **Grad-CAM++**.

Grad-CAM++ generates a visual representation of image regions associated with the model's selected prediction.

However:

> **Grad-CAM++ is not lesion segmentation.**

A heatmap can suggest that the model is attending to a particular region, but visual inspection alone is subjective.

Because the BUS-BRA dataset provides lesion masks, model attention could also be compared quantitatively with the documented lesion region.

---

# 📐 Quantitative Explainability Evaluation

Grad-CAM++ was evaluated across the complete **309-image test set**.

| Metric                       |     Result |
| ---------------------------- | ---------: |
| Mean IoU                     | **0.3397** |
| Mean Dice Score              | **0.4807** |
| Mean Attention Inside Lesion | **40.24%** |

These metrics provide different perspectives:

### IoU — Intersection over Union

Measures the overlap between the attention region and the reference lesion region.

### Dice Score

Another measure of spatial overlap between model attention and the lesion mask.

### Attention Inside Lesion

Measures how much of the model's attention falls within the documented lesion region.

Together, these metrics allow explainability to be evaluated quantitatively rather than relying only on visually appealing heatmaps.

---

# 🟢 Representative Strong Explanation

One representative test case achieved:

| Metric                  |     Result |
| ----------------------- | ---------: |
| IoU                     | **0.7292** |
| Dice                    | **0.8434** |
| Attention Inside Lesion | **64.09%** |

---

# 🔴 High-Confidence Failure Case

Explainability also revealed an important limitation.

One representative test case was:

* **Actually Benign**
* **Predicted as Malignant**
* Prediction Confidence: **99.91%**

Despite the incorrect classification, the attention comparison showed:

| Metric                  | Result |
| ----------------------- | -----: |
| IoU                     | 0.5283 |
| Dice                    | 0.6913 |
| Attention Inside Lesion | 50.18% |

This demonstrates an important point:

> **A model can attend substantially to the lesion and still make the wrong pathology prediction.**

For this reason, explainability should be considered a tool for investigation and model auditing—not proof that a prediction is clinically correct.

---

# 💡 What the Explainability Analysis Adds

The project therefore goes beyond:

```text
Image → Model → Benign / Malignant
```

and investigates:

```text
Image
  ↓
Classification
  ↓
Was the prediction correct?
  ↓
Generate Grad-CAM++
  ↓
Compare attention with lesion mask
  ↓
Measure overlap and lesion-focused attention
```

This makes the project not only a classification experiment, but also an exploration of **how explainable AI can be quantitatively evaluated in medical imaging**.

---

# 🌐 Interactive Streamlit Application

The final model was integrated into a Streamlit web application.

### Users can:

* Upload a breast ultrasound image
* Receive an AI-generated Benign or Malignant prediction
* View probability outputs
* Explore model separation
* View the probability distribution
* Generate a Grad-CAM++ visualization
* Explore representative BUS-BRA sample cases
* View available dataset reference information
* Compare model attention with reference lesion masks
* Download an analysis report

### Downloadable Analysis Report

The report includes:

* AI prediction
* Benign probability
* Malignant probability
* Model separation
* Grad-CAM++ explanation
* Important research and educational disclaimer
* Project information

---

# 🧭 Understanding the Results

The application also includes simple explanations for users unfamiliar with AI terminology.

It explains concepts such as:

### AI Prediction

The class selected by the model based on its learned patterns.

### Probability

The relative output values produced for Benign and Malignant classes.

### Model Separation

How clearly separated the two model outputs are.

A small difference between the two outputs may indicate a more borderline prediction.

### Grad-CAM++

A visual representation of image regions associated with the model's prediction.

### Reference Lesion Mask

A dataset-provided lesion annotation used for visualization and comparison.

It is important to distinguish this from the model's output:

> **The reference lesion mask is not generated by the AI model.**

---

# 🏥 From Prototype to Practice

This project explores a possible research direction for AI-assisted breast ultrasound analysis.

A conceptual workflow could be:

```text
Ultrasound Examination
        ↓
AI-Assisted Image Analysis
        ↓
Classification + Visual Explanation
        ↓
Professional Interpretation
```

The AI should remain a supportive analytical tool.

Any future real-world implementation would require:

* Independent external validation
* Evaluation across different institutions
* Different ultrasound equipment and acquisition protocols
* Clinical expert evaluation
* Prospective studies
* Safety assessment
* Appropriate regulatory review

---

# 🌐 Deployment

The project is deployed as an interactive web application.

| Component       | Platform     | Purpose                     |
| --------------- | ------------ | --------------------------- |
| Source Code     | GitHub       | `app.py` and `model.py`     |
| Model Weights   | Hugging Face | `Model_Best.pth` checkpoint |
| Web Application | Streamlit    | Interactive AI interface    |

The trained checkpoint is hosted separately because of its size.

### Deployment Flow

```text
GitHub Source Code
        ↓
Hugging Face Model Checkpoint
        ↓
Streamlit Application
        ↓
User Interaction
```

The core deployment files are:

```text
├── app.py
├── model.py
└── Model_Best.pth
```

---

# 🛠️ Technologies Used

**Python · PyTorch · Torchvision · NumPy · Pillow · Streamlit · Hugging Face · GitHub**

| Technology   | Role                                              |
| ------------ | ------------------------------------------------- |
| Python       | Core programming                                  |
| PyTorch      | Deep learning and inference                       |
| Torchvision  | EfficientNet-B3 architecture and image processing |
| NumPy        | Numerical processing                              |
| Pillow       | Image handling                                    |
| Streamlit    | Interactive web application                       |
| Hugging Face | Model checkpoint hosting                          |
| GitHub       | Source control and project hosting                |

### Core Architecture

> **EfficientNet-B3**

### Explainability Method

> **Grad-CAM++**

---

# ⚠️ Limitations

This project has several important limitations.

### 1. Research Dataset

The model was trained and evaluated using a research dataset.

Strong performance on the held-out test set does not guarantee equivalent performance in different clinical environments.

### 2. Binary Classification Only

The deployed model performs only:

> **Benign vs Malignant**

classification.

It does not represent a complete breast imaging diagnostic workflow.

### 3. High Confidence Does Not Guarantee Correctness

The confidence analysis showed that incorrect predictions could still receive high confidence.

Therefore, confidence should not be interpreted as clinical certainty.

### 4. Grad-CAM++ Is Not Segmentation

Grad-CAM++ visualizes model attention.

It does not provide a definitive lesion boundary and should not be interpreted as automated segmentation.

### 5. Not Clinically Validated

The current system should be considered a:

> **Research and educational proof-of-concept**

rather than a clinically validated diagnostic tool.

---

# 📝 Note on the Development Process

This project was developed as an end-to-end exploration of the intersection between **Medical Imaging and Artificial Intelligence**.

The work extended beyond training a single model.

It involved:

* Dataset verification
* Image-mask matching
* Multiple architecture experiments
* Data balancing strategies
* Lesion-focused approaches
* Mask-guided approaches
* Dual-view strategies
* Alternative classifiers
* Model comparison
* Validation-based threshold selection
* Untouched test evaluation
* Confidence analysis
* Calibration evaluation
* Image-level evaluation
* Case-level evaluation
* Grad-CAM++ generation
* Quantitative lesion-mask comparison
* Model checkpoint management
* Interactive Streamlit deployment

The final result represents an iterative development process rather than a single successful training run.

Some approaches performed well during development but were not selected after further evaluation.

That process itself was an important part of the project.

> **The goal was not simply to find the highest number, but to investigate which approach produced the strongest and most defensible final result.**

---

# ⚠️ Responsible Use & Disclaimer

This project is intended for:

* Research
* Education
* Demonstration
* Exploration of AI and explainable AI in medical imaging

It is **not intended for independent clinical diagnosis**.

AI-generated predictions may be:

* Incorrect
* Uncertain
* Overconfident

The model output should therefore not be used alone for:

* Diagnosis
* Treatment decisions
* Patient management

Grad-CAM++ represents model attention and should not be interpreted as definitive lesion segmentation or proof of clinical reasoning.

Before any real-world clinical implementation, the system would require independent external validation, clinical evaluation, safety assessment, and applicable regulatory review.

> **AI should support medical expertise—not replace it.**

---

# 🔗 Project Links

* 🌐 **Live Streamlit Application**
* 🤗 **Trained Model on Hugging Face**
* 💻 **GitHub Repository**

---

# 🙏 Acknowledgement

This project represents an independent exploration of how artificial intelligence and explainable AI can be applied to breast ultrasound image analysis.

The project was developed with a focus on not only measuring classification performance, but also investigating model attention and its relationship with documented lesion regions.

It reflects an interest in responsible and interpretable AI systems that may contribute to future research in medical imaging.

---

# 👩‍⚕️ About the Developer

## Zara Ashraf

**Medical Imaging Technologist**

**Medical Imaging · Artificial Intelligence · Explainable AI · Healthcare Technology · Research**

This project represents an independent exploration of how emerging AI technologies can be responsibly investigated and applied to challenges in medical imaging.

---

> **Designed & developed as an independent medical imaging AI project.**
