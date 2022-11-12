import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

modelTrained = [False]
bTraining = False #Is the model currently being trained

mnist = tf.keras.datasets.mnist
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
 
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

tfModel = tf.keras.models.Sequential()

tfModel.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
tfModel.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
tfModel.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
tfModel.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
