import streamlit as st
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

# ------------------------------------------------------
# 🔷 Project Description
# ------------------------------------------------------
st.title("✍️ Handwritten Digit Recognition App")
st.markdown("""
Welcome to the Handwritten Digit Recognition project!

This app demonstrates a simple machine learning model trained on the 
**Scikit-learn Digits Dataset**. You can:
- See how the model performs on test data
- Upload your own handwritten digit image (as JPG/PNG)
- Get real-time predictions!

The model is a Multi-layer Perceptron (MLP) classifier trained to recognize digits from 0 to 9.
""")

# ------------------------------------------------------
# 🔢 Load and Display Sample Digits
# ------------------------------------------------------
st.subheader("🔢 Sample Digits from Dataset")

# Load the digits dataset
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Show first 16 sample digits
def show_sample_digits():
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f"Label {digits.target[i]}")
        ax.axis('off')
    st.pyplot(fig)

show_sample_digits()

# ------------------------------------------------------
# 🧠 Train the Model
# ------------------------------------------------------
X_train, X_test = X[:1000], X[1000:]
y_train, y_test = y[:1000], y[1000:]

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic',
                    solver='sgd', learning_rate_init=0.1, tol=1e-4,
                    alpha=1e-4, random_state=1, verbose=False)

mlp.fit(X_train, y_train)

# ------------------------------------------------------
# ✅ Show Accuracy
# ------------------------------------------------------
st.subheader("📊 Model Accuracy")
predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.success(f"✅ Model Accuracy on Test Set: {accuracy:.2%}")

# ------------------------------------------------------
# 📤 Upload & Predict
# ------------------------------------------------------
st.subheader("📤 Upload Your Own Digit Image")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG of a digit)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to a larger square for uniform processing
    image = image.resize((100, 100), Image.Resampling.LANCZOS)

    # Convert to NumPy array
    image_np = np.array(image)

    # Thresholding: Convert to pure black & white
    image_np = np.where(image_np > 128, 255, 0).astype(np.uint8)

    # Find bounding box of the digit
    coords = np.column_stack(np.where(image_np < 255))
    if coords.size == 0:
        st.error("❌ Could not detect any digit in the image.")
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop to bounding box
        cropped = image_np[y_min:y_max+1, x_min:x_max+1]

        # Resize cropped image to 8x8
        digit_resized = Image.fromarray(cropped).resize((8, 8), Image.Resampling.LANCZOS)

        # Invert and scale like sklearn digits
        digit_array = np.array(ImageOps.invert(digit_resized)).astype(np.float64)
        digit_scaled = (digit_array / 255.0) * 16.0
        digit_flattened = digit_scaled.flatten().reshape(1, -1)

        # Prediction
        prediction = mlp.predict(digit_flattened)[0]
        st.image(digit_resized, caption="Processed Input (8×8)", width=150)
        st.success(f"🔢 Predicted Digit: {prediction}")
