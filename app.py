

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("models/model.h5")

# Waste categories (alphabetical order)
classes = [
    "Cardboard",
    "Glass",
    "Metal",
    "Organic",
    "Paper",
    "Plastic"
]

# Streamlit application
st.title("♻️ AI Waste Classification System")
st.write("Upload an image to identify its waste category.")

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # Image preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Waste"):

        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)
        category = classes[class_index]
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Category: {category}")

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.progress(float(confidence / 100))
