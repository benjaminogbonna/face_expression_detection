from django.shortcuts import render
import os
import numpy as np
from django.core.files.storage import default_storage
import tensorflow as tf
from .helper_function import load_and_prep_image

cur_dir = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join(cur_dir, 'model', 'model_3.h5'))

# Create your views here.

emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def index(request):
    if request.method == 'POST':        
        file = request.FILES["image"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        image = load_and_prep_image(file_url)        
        try:
            pred = model.predict(image)
            # prediction = emotion_list[int(np.argmax(pred))]
        except ValueError as e:
            prediction = 'An error occured, try again!'
        else:
             prediction = emotion_list[int(np.argmax(pred))]
        context = {
            'prediction': prediction,          
        }
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')
