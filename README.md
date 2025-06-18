# handwritten-digit-recognition
This project demonstrates a simple machine learning application that can recognize handwritten digits using the `scikit-learn` digits dataset (8×8 grayscale images).

It includes:

- ✅ A **Jupyter Notebook** to train and evaluate the model.
- ✅ A **Streamlit web application** where users can:
  - View sample digits
  - Upload a custom handwritten digit image (JPG or PNG)
  - Get real-time digit predictions from the trained model

---

### 🧠 Model Overview

- **Dataset**: `sklearn.datasets.load_digits`  
- **Model**: `MLPClassifier` (Multi-layer Perceptron)
- **Accuracy**: ~94–96% on unseen test data
- **Preprocessing**:
  - Grayscale conversion
  - Resizing and thresholding
  - Bounding box cropping
  - Scaling to match training format (8×8, 0–16 pixel intensity)

---

### 🎯 Key Features

- 📈 Interactive web app using **Streamlit**
- ✍️ Accepts real handwritten digit images
- 🔢 Predicts numbers from `0` to `9`
- ✅ Self-contained and beginner-friendly

---

## 🚀 Try the App Live

Click below to use the Streamlit app directly in your browser — no installation needed:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourname-streamlit-app.streamlit.app)

---
