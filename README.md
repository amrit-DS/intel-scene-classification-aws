# Intel Scene Classification with AWS SageMaker

End-to-end computer vision pipeline deployed on AWS SageMaker. Compares a baseline CNN trained from scratch against a ResNet50 transfer learning model for 6-class natural scene classification, with full MLOps practices including S3 data management, script-mode training jobs, and real-time inference endpoints.

**Key Results:** ResNet50 transfer learning improved test accuracy from **84.2% → 92.1%** over the baseline CNN, with both models trained and deployed on a budget of **under $0.50 USD**.

\---

## 📊 Results at a Glance

|Model|Test Accuracy|Input Size|Epochs|Training Time|Cost (USD)|
|-|-|-|-|-|-|
|Baseline CNN|**84.17%**|150×150|20|75 min|$0.14|
|ResNet50 (Transfer Learning)|**92.10%**|224×224|8|151 min|$0.29|

### Performance Comparison

**Baseline CNN — 84.2%**

!\[Baseline Confusion Matrix](results/baseline\_confusion\_matrix.png)

**ResNet50 Transfer Learning — 92.1%**

!\[ResNet50 Confusion Matrix](results/resnet50\_confusion\_matrix.png)

The transfer learning model showed the most significant improvement on the **glacier vs mountain** distinction, where the baseline CNN frequently confused the two classes due to similar snow-covered features.

\---

## 🎯 Project Overview

**Goal:** Classify natural and urban scenes into six categories — *buildings, forest, glacier, mountain, sea, street* — using deep learning, while comparing custom architectures vs. pre-trained networks and demonstrating production cloud deployment practices.

**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (\~17,000 training images, 3,000 test images, 6 balanced classes).

**Context:** Completed as part of MA5852 (Cloud Computing for Data Science) at James Cook University.

\---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
                    │     S3 Data Bucket      │
                    │  (train / val / test)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  SageMaker Script Mode  │
                    │      Training Jobs      │
                    │   (ml.m5.large × 2)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   S3 Model Artifacts    │
                    │      (.tar.gz)          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  SageMaker Endpoint     │
                    │  Real-time Inference    │
                    │    (HTTPS API)          │
                    └─────────────────────────┘
```

\---

## 🧠 Models

### 1\. Baseline CNN (trained from scratch)

A custom convolutional architecture designed to establish a performance floor without leveraging any pre-trained weights:

```
Input (150×150×3)
    → Conv2D(32) + ReLU → MaxPool
    → Conv2D(64) + ReLU → MaxPool
    → Flatten → Dense(128) + ReLU → Dropout(0.5)
    → Dense(6) + Softmax
```

Trained with Adam (lr=1e-3), 20 epochs, batch size 32, with on-the-fly augmentation (rotation ±20°, shifts, zoom, horizontal flips).

### 2\. ResNet50 Transfer Learning

A 50-layer pre-trained ResNet50 (ImageNet weights) with a custom classification head:

```
ResNet50 (frozen, ImageNet weights)
    → GlobalAveragePooling2D
    → Dropout(0.3)
    → Dense(6) + Softmax
```

Trained with Adam (lr=1e-4), 8 epochs (early stopped), batch size 32, ResNet50 standard preprocessing.

\---

## ☁️ AWS SageMaker Implementation

This project uses SageMaker's **script mode** rather than the SageMaker Studio GUI, demonstrating reproducible, production-grade ML workflows.

**What was implemented:**

* ✅ Custom training scripts (`train\_baseline.py`, `train\_resnet50.py`) deployed via the SageMaker TensorFlow estimator
* ✅ Hyperparameters passed at training time for reproducibility
* ✅ S3 data channels for training/validation
* ✅ Model artifact versioning and repackaging for TensorFlow Serving
* ✅ Real-time inference endpoint deployment
* ✅ Custom inference handler (`inference.py`) for image preprocessing

**Deployed infrastructure (verified — see screenshots):**

!\[SageMaker Training Job](docs/aws\_screenshots/05\_sagemaker\_training\_job.png)

*SageMaker training job completed successfully (account ID redacted for privacy).*

!\[SageMaker Endpoint Deployed](docs/aws\_screenshots/06\_sagemaker\_endpoint\_deployed.png)

*Real-time inference endpoint InService and operational.*

\---

## 💰 Cost Analysis

A deliberate cost-conscious design using `ml.m5.large` (CPU-based) instead of GPU instances:

|Resource|Hours|Rate|Cost|
|-|-|-|-|
|Baseline CNN training|1.25|$0.115/hr|$0.14|
|ResNet50 training|2.52|$0.115/hr|$0.29|
|**Total compute cost**|||**$0.43**|

**Trade-off:** GPU instances (`ml.p2.xlarge`) would have reduced training time by 60-80% but at \~7× the hourly cost — overkill for this dataset size.

\---

## 📁 Repository Structure

```
intel-scene-classification-aws/
├── notebooks/
│   └── intel\_scene\_classification.ipynb  # Main orchestration notebook
├── scripts/
│   ├── train\_baseline.py                 # SageMaker training script for baseline CNN
│   ├── train\_resnet50.py                 # SageMaker training script for ResNet50 TL
│   └── inference.py                      # Endpoint inference handler
├── results/
│   ├── baseline\_confusion\_matrix.png
│   ├── baseline\_accuracy\_curve.png
│   ├── baseline\_loss\_curve.png
│   ├── resnet50\_confusion\_matrix.png
│   └── model\_summary.csv
├── docs/
│   ├── project\_report.pdf                # Full academic report
│   └── aws\_screenshots/                  # AWS console evidence
├── requirements.txt
├── LICENSE
└── README.md
```

\---

## 🚀 Reproducing This Project

### Prerequisites

* AWS account with SageMaker access
* Python 3.10+
* AWS CLI configured

### Setup

```bash
# Clone the repository
git clone https://github.com/amrit-DS/intel-scene-classification-aws.git
cd intel-scene-classification-aws

# Install dependencies
pip install -r requirements.txt
```

### Running

1. Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) and upload to your S3 bucket.
2. Open `notebooks/intel\_scene\_classification.ipynb` in a SageMaker notebook instance.
3. Update the bucket name placeholders (`<ACCOUNT\_ID>`, `<BUCKET\_SUFFIX>`) with your AWS account values.
4. Run cells sequentially to train, evaluate, and deploy.

\---

## 📷 Training Curves

**Accuracy over epochs**

!\[Accuracy Curve](results/baseline\_accuracy\_curve.png)

**Loss over epochs**

!\[Loss Curve](results/baseline\_loss\_curve.png)

\---

## 📚 References

* He, K., et al. (2016). *Deep Residual Learning for Image Recognition* — [CVPR 2016](https://doi.org/10.1109/CVPR.2016.90)
* Krizhevsky, A., et al. (2012). *ImageNet Classification with Deep CNNs* — AlexNet
* Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* — JMLR
* AWS SageMaker Developer Guide — [docs.aws.amazon.com/sagemaker](https://docs.aws.amazon.com/sagemaker/)

\---

## 📫 Contact

**Amrit** — Aspiring Data Scientist | JCU Master's of Data Science

* LinkedIn: \[Amrit](https://www.linkedin.com/in/amrit-300668s/) 
* GitHub: [@amrit-DS](https://github.com/amrit-DS)

\---

*Built as part of MA5852 (Cloud Computing for Data Science), James Cook University, 2025.*

