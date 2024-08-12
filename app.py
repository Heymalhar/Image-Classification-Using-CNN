import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f'{working_dir}/Trained_fashion_mnist_model.h5'

print('successful')

model = tf.keras.models.load_model(model_path)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') # Convert to gray scale
    img_array = np.array(img)/255.0
    img_array = img_array.reshape((1, 28, 28, 1))

    return img_array

# Creating the streamlit app

st.title('Fashion Item Classifier')
st.write('Use this app if you are too lazy to identify the object yourself!')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)
    
    with col2:

        if st.button('Identify'):

            img_array = preprocess_image(uploaded_image)

            result = model.predict(img_array)

            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'The model predicted the class: {predicted_class}')
            
            st.success(f'Therefore, the object in the image is: {prediction}')