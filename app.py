from fastai.vision  import (
    load_learner,
)
from fastai.vision import *
import numpy as np
import glob
from random import shuffle
from PIL import Image
import urllib.request
import streamlit as st
from sklearn.metrics import cohen_kappa_score
import fastai.layers

#MODEL_URL = "https://github.com/nanarv/aibuildersdeploy-nana/raw/main/model/export.pkl"
#urllib.request.urlretrieve(MODEL_URL, "model/export.pkl")


def load_model():
    # Load the export.pkl file
    model = load_learner('model/')

    # Return the model for inference
    return model

def predict(image, model):
    # Perform inference using the loaded model
    predictions = model.predict(image)

    # Return the predictions
    return predictions

def quadratic_kappa(preds, targs):
    return cohen_kappa_score(preds, targs, weights='quadratic')


def main():
    # Set the title and description of the app
    st.title('Diabetic Retinopathy Detection AI (DRDAI) Prototype')
    st.write('This is a prototype for export.pkl, a diabetic retinopathy stage classification ai made using fastai version 1.0.61')

    # Load the model
    model = load_model()

    # Display an upload button for the user to upload an image
    uploaded_file = st.file_uploader('Upload your digital retinal image:', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)

        # Convert the PIL image to a fastai Image
        img_fastai = vision.Image(pil2tensor(image, np.float32).div_(255))
        
        # Perform inference and get the predictions
        predictions = predict(img_fastai, model)

        # Display the predictions
        stages = ["Undetected","Mild","Moderate","Severe","Proliferative DR"]
        for m in str(predictions[1]):
            if m.isdigit():
                st.write('DR Stage Prediction:', stages[int(m)])
                break
            else:
                pass

if __name__ == '__main__':
    main()
