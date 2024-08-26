import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

# Load the model
model = load_model('Fruitmodel.h5')

# Define your class names based on the model's output
class_names = ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'kiwi', 'pineapple']  # Replace with your actual class names

def predict_image(img):
    # Resize and preprocess the image
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(img_array)

    # Debugging: Print prediction shape and content
    st.write("Prediction shape:", prediction.shape)
    st.write("Prediction content:", prediction)

    try:
        # Handle prediction shape and get the predicted class index
        if prediction.ndim == 2:
            # Prediction is 2D: batch size x number of classes
            if prediction.shape[0] == 1:
                # Single image prediction
                predicted_index = np.argmax(prediction[0])
            else:
                # Multiple images prediction (unlikely here)
                predicted_index = np.argmax(prediction[0])
        elif prediction.ndim == 1:
            # Prediction is 1D: number of classes
            predicted_index = np.argmax(prediction)
        else:
            # Unexpected prediction shape
            st.write("Unexpected prediction shape")
            return "Error"

        # Map predicted index to class name
        predicted_class = class_names[predicted_index]
    except Exception as e:
        st.write("Error processing prediction:", e)
        return "Error"

    return predicted_class

def main():
    st.title("Fruit Classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        predicted_class = predict_image(img)
        st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
