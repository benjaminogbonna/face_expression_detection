import cv2
import numpy as np
import tensorflow as tf


# model_1 = tf.keras.models.load_model('forth_model.h5')
model_2 = tf.keras.models.load_model('model_7.h5')
emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
haarcascade = 'haarcascade_frontalface_default.xml'

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    face_cascade = cv2.CascadeClassifier(haarcascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        fc = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(fc, (48, 48)), -1), 0)
        prediction = model_2.predict(cropped_img)
        print(int(np.argmax(prediction)))
        max_index = int(np.argmax(prediction))
        cv2.putText(img, emotion_list[max_index], (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 179, 113), 2, cv2.LINE_AA)

    cv2.imshow('Image', cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
