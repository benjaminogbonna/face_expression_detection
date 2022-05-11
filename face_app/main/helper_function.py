import cv2
import os
import numpy as np

cur_dir = os.path.dirname(__file__)
haarcascade = os.path.join(cur_dir, 'model', 'haarcascade_frontalface_default.xml')


def load_and_prep_image(filename):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    image = cv2.imread(filename)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(haarcascade)

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)

        return cropped_img
