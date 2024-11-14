# 📹 Image Classification and Object Detection for Employee Monitoring in an Office Space

### Click on the image to see the YOLOv8 Tracking Demo Video
[![YOLOv8 Video Demo](https://img.youtube.com/vi/mcl4nsTSMms/0.jpg)](https://www.youtube.com/watch?v=mcl4nsTSMms)

## 👀 Project Overview
The CCTV-Office-Monitor project aims to classify and track employee behaviors in an office setting using a combination of machine learning and deep learning techniques. By analyzing office footage, the system detects actions like walking, sitting, talking, and falling to assess productivity, safety, and attendance patterns.
Since this project is part of a private group collaboration, only certain components covered in the article will be included in this repository.

### 🚀 Key Focus areas:

- ML: Scikit-learn (several classifiers)
- DL: ResNet18
- DL: Object Detection/Tracking (YOLOv8) 

## 📂 Dataset

The dataset, sourced from the University of Edinburgh, consists of **456,714 frames** collected from **four distinct office environments** (although only Office1(1-12 days) was considered in the project) over 20 days, capturing diverse employee behaviors. 

### 🗂️ Data Structure:
- **Resolution:** 1280x720 pixels (color images)
- **Class Labels:**
  - **Class 0**: Empty room
  - **Class 1**: Person standing/walking
  - **Class 2**: Person sitting
  - **Class 3**: Two or three people talking
  - **Class 4**: Person has fallen



## 🧠 Methodology

### 🏷️ 1. Traditional Machine Learning (Feature-Based Classification)
- **Features**: Extracted using **Img2Vec** (deep-learning vector representations) and **HOG** (shape-based gradients). These features are combined to enhance accuracy.
- **Models**: SVM, KNN, Decision Trees.., with **SVM** providing the best performance for behavior classification within the same office setting.
- **Goal**: Assess how well traditional ML can classify static behaviors based on engineered image features.

### 🤖 2. Deep Learning (ResNet-18)
- **Architecture**: ResNet-18, chosen for its effective handling of complex image patterns using residual connections (Good avoidance on vanashing gradients problem.)
- **Goal**: Evaluate deep learning’s capability for behavior recognition without extensive feature engineering.

### 📸 3. YOLOv8 Object Detection
- **Model**: YOLOv8, configured for real-time tracking of individuals with bounding boxes and behavior labels.
- **Goal**: Achieve real-time behavior tracking and assess model adaptability to different office environments.



## 📊 Key Results

### 1. Traditional Machine Learning (Feature-Based Classification)
- **Best Performance**: The **SVM** model using combined **Img2Vec + HOG** features provided the highest accuracy of **0.93** across same-office days, indicating that traditional ML models can be highly effective for behavior classification when using engineered features.

### 2. Deep Learning with ResNet-18
- **Validation Success but Environment Challenge**: **ResNet-18** achieved strong validation accuracy (**0.9379**), but struggled with different-office environments, indicating a lack of adaptability without further training or adaptation.

### 3. YOLOv8 for Object Detection
- **Real-Time Tracking Limitation**: **YOLOv8** performed well within the trained office setting (accuracy **0.9405**), but accuracy drastically dropped in different office settings (to **0.031**), highlighting domain transfer challenges.



# Further improvements 💭

### Model Training and Optimization 🚀
- **Extend Training for DL Models**: Increase training epochs to boost accuracy on untrained days.
- **Optimize Hyperparameters**: Fine-tune parameters like learning rate and batch size for better results.
- **Apply Regularization Techniques**: Use methods like dropout and batch normalization for better generalization.
- **Explore Advanced DL Models**: Evaluate architectures such as EfficientNet or others for potential accuracy gains.
- **Try More Classifiers**: Test models like gradient boosting for ML alternatives.

### Data and Features Enhancements 📊
- **Add Temporal Diversity**: Include data from more days to reduce overfitting to specific scenarios.
- **Expand to Multiple Offices**: Use data from different office environments to improve generalization.
- **Incorporate Temporal/Spatial Features**: Add features related to time and position for improved predictions.
- **Refine Feature Sets for ML**: Prioritize relevant features with a feature selection process for efficiency.
- **Data Augmentation**: Introduce varied backgrounds and lighting to make training data more diverse.

### User Experience and Deployment 🖥️
- **Optimize for Live Processing**: Improve model speed to enable real-time monitoring.
- **Develop a User Interface**: Create a UI for live tracking and activity insights.

***⚠️ NOTE: This project was conducted on Google Colab using paid limited resources. Additional computing power could significantly improve performance and outcomes.***

---

# Delving deeper into the files in this repo. 🗂️

- Taking into account what was mentioned in the section **"👀 Project Overview"** here we delve deeper into an explanation of the files that are provided in this repository

# 📦 Dependencies and Imports for the ResNet (AdaptedForResNet)

The following Python libraries were essential for the implementation of this project. Each of them played a significant role in various stages of the pipeline, such as data processing, modeling, visualization, and performance evaluation:

```python
import imp
import os
import rarfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from google.colab import drive
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import datasets, transforms, models
```

### 📅 daysCSV
A public dataset used for analysis and training in the project.

### 📂 set_separate
- Contains frames separated into training and validation sets:
- **Train** and **Val**: Each folder has subfolders labeled `0-4`, which represent different behaviors observed in the frames.

### 📊 Resnet10epochs and Resnet100epochs
- Contains evaluation results for days `6` and `12`:
- **Confusion Matrices** and **Results**: Performance metrics for each training epoch (10 and 100).

### 📝 Annotations YOLOV8 "Proof of Concept" - YOLOV8_annotations_Proof_of_Concept
Annotations created using YOLOV8 as a proof of concept for object detection model testing.

### 🎓 Trained ResNet_epochs (.pth)
Trained ResNet models (10 and 100 epochs), stored for performance evaluation and further analysis.




