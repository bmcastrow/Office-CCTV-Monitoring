# 📹 CCTV-Office-Monitor

## 👀 Project Overview
The CCTV-Office-Monitor project aims to classify and track employee behaviors in an office setting using a combination of machine learning and deep learning techniques. By analyzing office footage, the system detects actions like walking, sitting, talking, and falling to assess productivity, safety, and attendance patterns.
Although 3 areas were explored and mentioned in the article because of confidentiality purposes, only certain files will be included in this repo.

### 🚀 Key Focus areas:

- ML: Scikit-learn (several classifiers)
- DL: ResNet18
- DL: Object Detection/Tracking (YOLOv8) 

## 📂 Dataset

The dataset, sourced from the University of Edinburgh, consists of **456,714 frames** collected from **four distinct office environments** (although only office 1 was considered in the project) over 20 days, capturing diverse employee behaviors. 

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
- **Architecture**: ResNet-18, chosen for its effective handling of complex image patterns using residual connections, avoiding vanashing gradients problem.
- **Goal**: Evaluate deep learning’s capability for behavior recognition without extensive feature engineering.

### 📸 3. YOLOv8 Object Detection
- **Model**: YOLOv8, configured for real-time tracking of individuals with bounding boxes and behavior labels.
- **Goal**: Achieve real-time behavior tracking and assess model adaptability to different office environments.

## 📊 Results & Analysis

This section showcases key visuals from the study, illustrating model performance across different methods.

### 🔬 SVM Performance (Traditional ML)
Confusion matrices showing SVM’s classification results across different days using Img2Vec features. 

![SVM Confusion Matrix](link_to_svm_confusion_matrix_image)

---

### 🧠 ResNet-18 (Deep Learning)
Training metrics for ResNet-18 at 100 epochs, comparing training and validation accuracy and loss.

![ResNet Training Metrics](link_to_resnet_metrics_image)

---

### 🎥 YOLOv8 Object Detection
Sample output frame from YOLOv8 tracking on Day 6, with bounding boxes around individuals to indicate real-time detection and labeling.

![YOLOv8 Detection Output](link_to_yolo_detection_output_image)

![YOLOv8 Tracking Demo GIF](link_to_your_gif.gif)
#### YOLOv8 Tracking Demo Video
[![YOLOv8 Video Demo](https://img.youtube.com/vi/mcl4nsTSMms/0.jpg)](https://www.youtube.com/watch?v=mcl4nsTSMms)

---

### 📊 Comparative Performance Summary
Performance comparison for ML (SVM), ResNet, and YOLOv8 models across-validation, same-office, and cross-office data.

![Performance Comparison Table](link_to_performance_comparison_image)



