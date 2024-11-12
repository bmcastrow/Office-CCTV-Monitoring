# 📹 CCTV-Office-Monitor

## 👀 Project Overview
The CCTV-Office-Monitor project aims to classify and track employee behaviors in an office setting using a combination of machine learning and deep learning techniques. By analyzing office footage, the system detects actions like walking, sitting, talking, and falling to assess productivity, safety, and attendance patterns.

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





