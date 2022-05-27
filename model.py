# modules for data manipulation and data preprocessing
import pandas as pd
import numpy as np
from numpy import asarray

# modules for iamge processing
import os
from tqdm.notebook import tqdm
from PIL import Image

# modules for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# modules to create, train and save the model
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

# to ignore the warnings syntax generated
import warnings
warnings.filterwarnings('ignore')

# set the directory of the dataset (directory path may differ)
BASE_DIR = '../UTKFace/'

# create array to store the image filename with their corresponding age and gender
image_paths = []
age_labels = []
gender_labels = []

# for each image filename, we split them into their path, age and gender
for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename) # assign filename to image path
    temp = filename.split('_')                    # split the filename with "_"
    age = int(temp[0])                            # get the age
    gender = int(temp[1])                         # get the gender
    image_paths.append(image_path)                # store image path into its array
    age_labels.append(age)                        # store age into its array
    gender_labels.append(gender)                  # store gender into its array

# convert those array into a dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# this is a function to preprocess the image 
def extract_features(images):
    features = []                                             # create a new array for the processed image
    for image in tqdm(images):
        img = load_img(image, grayscale=True)                 # load the image as grayscale
        img = img.resize((128, 128), Image.ANTIALIAS)         # resize the image and apply antialias
        img = np.array(img)                                   # get the pixels of image
        features.append(img)                                  # store the pixel
        
    features = np.array(features)                             # make the array a NumPy array
    features = features.reshape(len(features), 128, 128, 1)   # reshape the array for the model
    return features

X = extract_features(df['image'])

X = X/255.0                         # normalize the input
y_age = np.array(df['age'])         # prepare training output for age  
y_gender = np.array(df['gender'])   # prepare training output for gender

# creation of model begins here
inputs = Input((128, 128, 1)) # set the input size

# create 4 convolutional layers followed by maxpooling
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

# flattens the multi-dimensional input tensors into a single dimension
flatten = Flatten() (maxp_4)

# create 2 seperate fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

# set dropout rate for neuron in each layer to prevent overfitting
dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

# set the output layer activation function as sigmoid for gender and ReLu for age
output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

# create the model with input and outputs
model = Model(inputs=[inputs], outputs=[output_1, output_2])

# compile the result of loss function and the metrics of each training process
model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['acc'])

# train model with preferable batch size, epochs and validation set split ratio
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=15, validation_split=0.2)

# save model and the history of each epoch
tf.keras.models.save_model(model,'my_model.hdf5')
history = pd.DataFrame(history.history)
history.to_csv('history.csv')
