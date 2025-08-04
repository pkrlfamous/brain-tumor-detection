# Brain Tumor Detection App

This is a simple web application built with Streamlit that allows users to upload a brain MRI image and get a prediction on whether a tumor is present or not. The model used for prediction is a pre-trained Keras `.h5` model.

## Features

- Upload MRI image (JPG or PNG)
- Get prediction using a deep learning model
- Visualize the uploaded image and model's focus using Grad-CAM

## Getting Started

1. Clone the repository.
2. Install dependencies after making conda env
3. Run the app: `streamlit run app.py`


## Files

- `app.py` – Main Streamlit app file
- `model.h5` – Pre-trained model file
- `requirements.txt` – Python dependencies
- `brain-tumor-detection` – jupyter notebook
- `brain_mri_dataset` - dataset

## Deployment

The app can be deployed for free using [Streamlit Cloud](https://streamlit.io/cloud).
