import streamlit as st
from PIL import Image
import tensorflow as tf
import joblib
import pickle

from tensorflow.keras.layers import Layer
import numpy as np
    

# model = tf.keras.models.load_model('satellite_military.h5')
# model = pickle.load(open('satellite_military_flutter.pkl', 'rb'))

model_path = 'satellite_military_flutter.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

st.set_page_config(page_title="Aircraft Detection", page_icon="✈️")


st.title("Aircraft Detection")


# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a Streamlit app
st.title('TFLite Model Demo')

# Upload an image for inference
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = uploaded_image.read()
    # image = image.astype(np.float32)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image (you might need to adapt this based on your model)
    # For example, resize the image to the input shape expected by the model
    input_shape = input_details[0]['shape']
    # image = image / 255.0  # Normalize if necessary
    image = np.array(Image.open(uploaded_image).resize((input_shape[1], input_shape[2])))
    image = image.astype(np.float32) / 255.0

    class_labels = ['A1','A2','A3','A4','A12','A6','A7','A8','A9','A10','A11','A5','A13','A14','A15','A16','A17','A18','A19','A20']  
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image.reshape(input_shape))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data)

    # Get the label name for the predicted class index
    predicted_label = class_labels[predicted_class_index]

    # Display the predicted label
    st.write(f'Prediction: {predicted_label}')


