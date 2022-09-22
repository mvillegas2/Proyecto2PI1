from cgitb import handler
from importlib.resources import path
from django.shortcuts import render
import numpy as np

from .models import mlModels

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

# Create your views here.


def home(request):
    petclassiffierFiles = mlModels.objects.filter(priority=1)[0]
    path_arch = petclassiffierFiles.architecture.path
    path_weights = petclassiffierFiles.weights.path
    
    with open(path_arch) as json_file:
        json_config = json_file.read()
    
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(path_weights)

    if request.method == 'POST':
        handle_uploaded_file(request.FILES['sentFile'])
        image = tf.keras.preprocessing.image.load_img('static/test.jpg', target_size=(150, 150, 3))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        pred = tf.keras.activations.sigmoid(model.predict(input_arr))[0][0]
        caption = f'dog prob {pred}, cat prob {1-pred}'
        return render(request, 'home.html', {'caption': caption})

    return render(request, 'home.html')


def handle_uploaded_file(f):
        with open('static/test.jpg', 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)


