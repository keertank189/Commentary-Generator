import keras 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.cross validation import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

v_df = pd.read_csv("video_data.csv")
print(v_df.head())

labels = v_df.iloc[:,-1]
print(labels)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(v_df.drop('4185',axis = 1), dtype = float))
mapu = []
for i in range(len(labels)):
    if labels[i]=='3pointer':
       mapu.append(['3point',3])
    elif labels[i]=='2pointer':
       mapu.append(['2point',2])
    else:
       mapu.append(['miss',0])

       
num_channel=1
num_of_samples=150
label = np.ones((num_of_samples,),dtype='int64')
lengths=[50,100,150]

print(lengths)

for k in range(0,lengths[0]):
    label[k]=mapu[0][1]


i=1
while(i<3):
     for j in range(lengths[i-1],lengths[i]):
         label[j]=mapu[i][1]
     i+=1



Y= np_utils.to_categorical(label,3)
x,y = shuffle(X,Y, random_state=5)

#Split the dataset
X_train,X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=4)
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=4) 

#Defining the model
shape=X_train[0].shape
print(shape)

model = Sequential()
model.add( Conv2D(64, (1,1), input_shape=shape, name="input"))
#model.add(Dropout(0.33))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))


model.add( Conv2D(64, (1,1)))
#model.add(Dropout(0.33))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
#model.add(Dropout(0.33))
model.add(Dense(3,activation='softmax',name='op'))

adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'C:/collegework/CDSAML/training/weights-improvement-pooled-mono-312-frames-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=75, batch_size=16, callbacks=callback_list, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('the testing accuracy is',score[1])
test_image = X_test

(model.predict(test_image))
print(model.predict_classes(test_image))
