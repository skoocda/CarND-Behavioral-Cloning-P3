import argparse
import base64
import cv2
import numpy as np
from datetime import datetime
import os
import shutil
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras import __version__ as keras_version
from keras.models import model_from_json, load_model
import h5py
import tensorflow as tf
# Fix error with Keras and TensorFlow
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
steering_history = 0.0

def preprocess_image(img):
    '''
    Method for preprocessing images
    '''
    out = img[50:140, :, :]
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.resize(out, (200, 66), interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2YUV)
    return out

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global steering_history
        steering_angle = data["steering_angle"]
        throttle = data["throttle"]
        speed = data["speed"]
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        img = preprocess_image(image_array)
        transformed_image_array = img[None, :, :, :]
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
        blur = 0.5
        steering_output = (steering_angle * blur) + (steering_history * (1-blur))
        steering_history = steering_angle
        speed_max = 32. #ish
        speed_map = float(speed)/speed_max #maps to [0,1]
        throttle = 1. / speed_map #inverse, accelerate when slow
        #throttle = 0.25
        #if float(speed) < 10:
        #    throttle = 1
        print(steering_angle, throttle)
        send_control(steering_output, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('image_folder', type=str, nargs='?', default='',help='Path to folder for saving images.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())
    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)