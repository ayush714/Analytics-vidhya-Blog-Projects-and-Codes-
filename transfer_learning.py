# Step 1:-  Importing the necessary libraries  

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

# Step 2:- Loading and dividing the data  
(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

# Step 3:- Preprocess the image 
def preprocess_image_input(input_images):
   input_images = input_images.astype('float32')
   output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
   return output_ims
train = preprocess_image_input(training_images)
test = preprocess_image_input(validation_images)

# Step 4:- Building the Model 
def load_params(inputs): 
   '''
   Function for loading the weights of ResNet with the weight of imagenet.
   '''
   feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                              include_top=False,
                                              weights='imagenet')(inputs)
   return feature_extractor
def classifier(inputs):
   '''
   Classifier which we will use for fine tuning.
   '''
   x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
   x = tf.keras.layers.Flatten()(x)
   x = tf.keras.layers.Dense(1024, activation="relu")(x)
   x = tf.keras.layers.Dense(512, activation="relu")(x)
   x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
   return x
def final_model(inputs):
   '''
   Final model for calling our functions
   '''
   resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs) #  We will upsample our image from 32 x 32 x 3 to 224 x 224 x 3
   resnet_weights = load_params(resize) # we are loading the weights
   classification = classifier(resnet_weights) # simple classifier that we made
   return classification
def compile_model():
   inputs = tf.keras.layers.Input(shape=(32,32,3))
   classification = final_model(inputs)
   model = tf.keras.Model(inputs=inputs, outputs = classification) # stacking up our model on top of Resnet model
   model.compile(optimizer='SGD',
               loss='sparse_categorical_crossentropy', # this loss is for multi-class classification
               metrics = ['accuracy'])
   return model
model = compile_model()
model.summary()

# Step 5 :-  Training the model 
history = model.fit(train_X, training_labels, epochs=5, validation_data = (valid_X, validation_labels)