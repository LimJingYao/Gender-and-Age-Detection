# import the libraries
pip install tensorflow
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# load the model
model = tf.keras.models.load_model('my_model.hdf5')

# create a function to predict the age and gender of an image
def import_and_predict(photo, model):
        photo = photo.resize((128,128))    # resize the image to fit into the model
        photo = photo.convert(mode='L')    # convert the image to grayscale
        photo = np.asarray(photo)          # get the pixels of the image
        photo = photo.reshape(128,128,1)   # reshape the pixel of the image
        photo = photo/255.0                # normalize the pixel into smaller numbers
        
        # predict the age and gender of the image
        pred = model.predict(photo.reshape(1, 128, 128, 1)) 
        
        # create a dictionary for the gender label
        gender_dict = {0:'Male', 1:'Female'}
        pred_gender = gender_dict[round(pred[0][0][0])]   # get predicted gender label
        pred_age = round(pred[1][0][0])                   # get predicted age
        st.write("Predicted Gender:", pred_gender)
        st.write("Predicted Age:", pred_age)
        
# create the interface with title, header and a section to upload the image
st.title("Age and Gender Detection")      
st.header("This is an web app to predict age and gender from a human face image.")
file = st.file_uploader("Upload an image file", type=["jpg", "png"])
        
if file is None:
    st.text("Please upload an image file ^^")
else:
    image = Image.open(file)              # load the image file
    st.image(image, width=300)            # show the image with specific width
    import_and_predict(image, model)      # run the prediction
