import keras 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from keras.layers import Conv2D
from keras.layers.core import Dense,Activation,Flatten,Dropout
warnings.filterwarnings('ignore')

v_df = pd.read_csv("video_data.csv")
print(v_df.head())

#print(v_df['4162'])
labels = v_df.iloc[:,-1]
print(labels)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
#print(y)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(v_df.drop('4185',axis = 1), dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("video features:")
print((X_test[1]))
print("coorect output:")
print(y_test[1])

import tensorflow as tf
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(3))
model.add(layers.Activation(tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=15,
                    batch_size=4)

test_loss_ann, test_acc_ann = model.evaluate(X_test,y_test)
predictions_ann = model.predict(X_test)


del model
print("Predicted output:")
print(np.argmax(predictions_ann[1]))
print(test_acc_ann)