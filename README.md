# ✍️ Handwritten Digit Recognition using Scikit-learn

This project demonstrates a simple machine learning application that can recognize handwritten digits using the Scikit-learn `digits` dataset (8×8 grayscale images).

---

## 🚀 Try the App Live

Click below to use the Streamlit app directly in your browser — no installation needed:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://handwritten-digit-recognition-6sbeyjtizbqsp6w9zhlgq4.streamlit.app/)

---

## 📁 Project Overview

This repo contains:

| File | Description |
|------|-------------|
| `app.py` | Streamlit app that allows users to upload and classify digit images |
| `Recognizing handwritten digits in Scikit Learn.ipynb` | Jupyter notebook for model training and evaluation |
| `requirements.txt` | Dependencies for running the app |
| `sample_digit1.png`, `sample_digit2.png` | Sample digit images to test the app |
| `README.md` | Project overview and usage instructions |

---

## 🧠 Model Info

- **Dataset**: `sklearn.datasets.load_digits` (8×8 grayscale)
- **Model**: `MLPClassifier` from Scikit-learn
- **Accuracy**: ~94–96% on test data
- **Preprocessing**:
  - Grayscale conversion
  - Thresholding & bounding box cropping
  - Resizing to 8×8
  - Scaling pixel values to match training format

---

## 🖼 Sample Test Images

Example test images available in this repo:

| Sample | 
|--------|
| sample_digit1.png | 
| sample_digit2.png | 

You can upload them into the app to test predictions.

---

## 🛠 How to Run Locally

Follow these steps to run the project on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/VarshiniReddy9/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app1.py

# 5. (Optional) Run the Jupyter Notebook
jupyter notebook
# Then open 'Recognizing handwritten digits in Scikit Learn.ipynb'

```
---

## 👤 Author

**Varshini Reddy**  
🔗 [GitHub Profile](https://github.com/VarshiniReddy9)
