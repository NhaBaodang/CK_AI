import numpy as np
from keras_preprocessing import image
import cv2
import os
from keras.models import Sequential, load_model
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
model = Sequential()
new_model = load_model('face1.h5')


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



