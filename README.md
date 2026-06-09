# MobileNet-Based Image Classification on CIFAR-10 Using a Custom Focal Loss Function

## Project Title

**Image Classification on CIFAR-10 Using MobileNet Architecture and Custom Focal Loss Function**

---

# Team Members

| S.No | Name                      | Roll / ID   |
| ---- | ------------------------- | ----------- |
| 1    | Bhargav Prasanna Upputuri | AD25B1037 |
| 2    | Teja Abhiram Rongala      | CS25B1033 |
| 3    | Deepak Satya Paluri       | CS25B1031 |

---

# Project Overview

This project presents an image classification system using the CIFAR-10 dataset and the MobileNet deep learning architecture. The system leverages transfer learning with MobileNet and incorporates a custom Focal Loss function to improve classification performance.

The project aims to classify images into ten predefined categories while evaluating performance using Accuracy, Precision, Recall, and F1-Score. The model is trained using TensorFlow and Keras and executed using GPU acceleration through Google Colab.

---

# Problem Statement

Image classification is one of the fundamental tasks in computer vision. Traditional convolutional neural networks often require significant computational resources and training time. MobileNet provides an efficient alternative by utilizing depthwise separable convolutions while maintaining strong classification performance.

The objective of this project is to build an efficient image classification model capable of accurately classifying CIFAR-10 images using MobileNet and a custom loss function.

---

# Project Objectives

* Perform image classification on the CIFAR-10 dataset.
* Utilize MobileNet as the base architecture.
* Implement a custom Focal Loss function.
* Evaluate model performance using multiple metrics.
* Generate visualizations for model analysis.
* Save the trained model for future deployment.

---

# Dataset Description

## CIFAR-10 Dataset

The CIFAR-10 dataset contains 60,000 RGB images distributed across 10 classes.

### Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Statistics

| Category          | Count       |
| ----------------- | ----------- |
| Training Images   | 50,000      |
| Testing Images    | 10,000      |
| Number of Classes | 10          |
| Image Size        | 32 × 32 × 3 |

---

# Technologies Used

## Programming Language

* Python

## Libraries

* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-Learn
* Pickle

## Development Tools

* Google Colab
* GitHub
* Visual Studio Code

---

# Model Architecture

## MobileNet

MobileNet is a lightweight convolutional neural network specifically designed for efficient image classification. It reduces computational cost through depthwise separable convolutions.

### Architecture Components

* Pretrained MobileNet Base
* Global Average Pooling Layer
* Dropout Layer
* Dense Classification Layer
* Softmax Activation Function

---

# Custom Loss Function

## Focal Loss

The project uses Focal Loss function

### Advantages

* Focuses training on hard-to-classify examples.
* Reduces the effect of easy samples.
* Improves model robustness.
* Enhances classification performance.

---

# Project Structure

```text
ICSProject/
│
├── Train.py
├── custom_loss.py
├── utils.py
├── requirements.txt
├── README.md
│
├── cifar-10-batches-py/
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
│
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── saved_model/
    └── mobilenet_cifar10.keras
```

---

# Team Responsibilities

To ensure effective collaboration and project completion, responsibilities were distributed among the team members as follows:

## Bhargav Prasanna Upputuri

### Responsibilities

* Project planning and repository management.
* MobileNet model implementation.
* Integration of TensorFlow and Keras components.
* Training and testing the model on Google Colab.
* Model saving and deployment preparation.
* GitHub version control management.

### Deliverables

* MobileNet architecture implementation.
* Training pipeline.
* Model storage and execution workflow.
* GitHub repository maintenance.

---

## Teja Abhiram Rongala

### Responsibilities

* Dataset organization and preprocessing.
* CIFAR-10 batch loading implementation.
* Data normalization and resizing.
* Data validation and preparation for training.
* Utility module development.

### Deliverables

* Dataset preparation workflow.
* Data preprocessing pipeline.
* Utility functions for dataset loading.
* Dataset documentation.

---

## Deepak Satya Paluri

### Responsibilities

* Custom Focal Loss implementation.
* Performance evaluation and metric calculation.
* Accuracy, Precision, Recall, and F1-Score analysis.
* Confusion Matrix generation.
* Visualization and report preparation.

### Deliverables

* Custom loss function implementation.
* Performance evaluation module.
* Graph generation.
* Documentation and report support.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/bhargavprasanna/ICSProject.git
```

Navigate into the project directory:

```bash
cd ICSProject
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute the training script:

```bash
python Train.py
```

The system will:

1. Load the CIFAR-10 dataset.
2. Preprocess images.
3. Train MobileNet.
4. Apply custom Focal Loss.
5. Evaluate performance.
6. Generate visualizations.
7. Save the trained model.

---

# Evaluation Metrics

The following metrics are used to assess model performance:

## Accuracy

Measures the percentage of correctly classified images.

## Precision

Measures the correctness of positive predictions.

## Recall

Measures the model's ability to identify relevant samples.

## F1-Score

Provides a balanced evaluation using both precision and recall.

---

# Generated Outputs

## Results Directory

* Accuracy Graph
* Loss Graph
* Confusion Matrix
* Metrics Report

## Saved Model

* Trained MobileNet Model (`mobilenet_cifar10.keras`)

---

# Learning Outcomes

This project provided practical experience in:

* Deep Learning Fundamentals
* Convolutional Neural Networks
* Transfer Learning
* MobileNet Architecture
* Custom Loss Functions
* Image Classification
* Performance Evaluation Metrics
* GPU-Based Training
* Git and GitHub Collaboration

---

# Future Enhancements

* Fine-tuning MobileNet layers.
* Hyperparameter optimization.
* Data augmentation techniques.
* Comparison with ResNet and EfficientNet.
* Web-based deployment.
* Real-time image classification applications.

---

# Conclusion

This project successfully demonstrates an efficient image classification framework using MobileNet and a custom Focal Loss function on the CIFAR-10 dataset. The system achieves robust performance while maintaining computational efficiency and provides a strong foundation for future computer vision applications.

---

# Acknowledgements

We express our gratitude to our faculty members, institution, TensorFlow developers, CIFAR-10 dataset creators, and the open-source community for their valuable resources and support throughout the project.

---

# Authors

**Bhargav Prasanna Upputuri**


**Teja Abhiram Rongala**


**Deepak Satya Paluri**
