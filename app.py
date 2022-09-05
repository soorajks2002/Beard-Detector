import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

@st.cache(allow_output_mutation = True) 
def get_model():
    path = 'beard_classification'
    return tf.keras.models.load_model(path)

model = get_model()

output = {0 : 'BEARDLESS',1 : 'BEARD'}

st.title('BEARD DETECTOR')

uploaded_file = st.file_uploader("Choose A Image")

if uploaded_file is not None :
    img = Image.open(uploaded_file)
    img = img.resize((224,224))
    
    img = np.asarray(img)
    img = np.resize(img,(1,224,224,3))
    img = img/225
    
    pred = np.argmax(model.predict(img))
    st.header(output[pred])