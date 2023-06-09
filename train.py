import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils, to_categorical, load_img, img_to_array
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from keras_preprocessing import image

from sklearn import preprocessing
from sklearn.utils import validation
from sklearn.model_selection import train_test_split


data='C:\\Users\\admin\\Downloads\\AI_FN\\Face1'
Data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
Train = Data.flow_from_directory(data, target_size=(150,150),batch_size=32, class_mode='categorical')
Train.class_indices

model=Sequential()
model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=(150,150,3))) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same')) 
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same')) 
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(13,activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train=model.fit(Train, epochs=50, batch_size=256, verbose=1)
model.save('C:\\Users\\admin\\Downloads\\AI_FN\\face.h5')
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
model = Sequential()


new_model = load_model('face.h5')

while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\Users\\admin\\Downloads\\AI_FN\\Test'+ str(i) + ".jpg", frame)
    test_image = image.load_img('C:\\Users\\admin\\Downloads\\AI_FN\\Test' + str(i) + ".jpg", target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image=test_image.astype('float32')
    test_image = np.expand_dims(test_image, axis=0)
    result = (new_model.predict(test_image).argmax())
    classes = ['Anh Việt', 'Bảo Nha', 'Chí Nhân', 'Đoàn Quang Nhat', 'Giang Hà', 'Hải Hiếu', 'Kim Thoa', 'Lê Quang Nhat', 'Sơn Lâm', 'Tấn Vũ', 'Thảo Hưng', 'Thiên Phát', 'Tuấn Nam']

    print('Đây là : {}'.format(classes[result]))
    os.remove('C:\\Users\\admin\\Downloads\\AI_FN\\Test' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
