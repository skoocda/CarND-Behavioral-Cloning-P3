import math
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from os import getcwd
import csv
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
tf.python.control_flow_ops = tf

def init_model():
    '''
    Define the model and return for training
    Convnet for behavioral cloning
    Inspired by Nvidia and comma.ai
    Deprecated, not actually called in main() anymore 
    Just here to view the architecture quickly!
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(32, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(8, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(1))
    return model

def plot_histogram(angles, hist, center, width, spbin):
    '''
    Utility method to show a histogram of data
    '''
    plt.figure()
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (spbin, spbin), 'k-')
    plt.show()


def normalize_data(images, angles, do_plot=False):
    '''
    Eliminate some useless straight line driving data
    '''
    num_bins = 37
    avg_samples_per_bin = len(angles)/num_bins
    target = avg_samples_per_bin / 2
    hist, bins = np.histogram(angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    if do_plot:
        plot_histogram(angles, hist, center, width, avg_samples_per_bin)

    keep_prob = []
    for i in range(num_bins):
        if hist[i] < target:
            keep_prob.append(1.)
        else:
            keep_prob.append(1./(hist[i]/target))

    remove_list = []
    for m in range(len(angles)):
        for n in range(num_bins):
            if angles[m] > bins[n] and angles[m] <= bins[n+1]:
                if np.random.rand() > keep_prob[n]:
                    remove_list.append(m)

    images = np.delete(images, remove_list, axis=0)
    angles = np.delete(angles, remove_list)

    hist, bins = np.histogram(angles, num_bins)
    if do_plot:
        plot_histogram(angles, hist, center, width, avg_samples_per_bin)
    return images, angles

def displayCV2(img):
    '''
    Utility method to display a CV2 Image
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_dataset(X, y, y_pred=None):
    '''
    method to format image prior to displaying.
    Converts to BGR, displays steering angle and frame number
    Adds lines for observed and predicted steering angle to image.
    '''
    font = cv2.FONT_HERSHEY_TRIPLEX
    clr = (200, 100, 100)
    for i in range(len(X)):
        image = X[i]
        angle = y[i]
        pred_angle = y_pred[i] if y_pred is not None else 0
        img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[0:2]
        # apply text for frame number and steering angle
        cv2.putText(img, 'frame: '+str(i), org=(2, 18), fontFace=font,
                    fontScale=.5, color=clr, thickness=1)
        cv2.putText(img, 'angle: '+str(i), org=(2, 33), fontFace=font,
                    fontScale=.5, color=clr, thickness=1)
        # apply a line representing the steering angle
        cv2.line(img, (int(w/2), h), (int(w/2+angle*w/4), int(h/2)), (0, 255, 0), thickness=4)
        if abs(pred_angle) > 0:
            cv2.line(img, (int(w/2), h), (int(w/2+pred_angle*w/4), int(h/2)), (0, 0, 255), thickness=4)
        displayCV2(img)
        

def preprocess_image(img):
    '''
    Method for preprocessing images, as per nVidia paper
    '''
    out = img[50:140, :, :]
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.resize(out, (200, 66), interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2YUV)
    return out

def random_distort(img):
    '''
    Method for brightness adjust + vertical shift
    '''
    img1 = img.astype(float)
    img2 = random_brightness(img1)
    img3 = partial_shadow(img2)
    img_ = horizon_shift(img3)
    return img_.astype(np.uint8)

def random_brightness(img):
    '''
    Random brightness - the mask bit keeps values from going beyond (0,255)
    '''
    out_img = img.astype(float)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (out_img[:, :, 0] + value) > 255 
    else:
        mask = (out_img[:, :, 0] + value) < 0
    out_img[:, :, 0] += np.where(mask, 0, value)
    return out_img

def partial_shadow(img):
    '''
    Random shadow - full height, either left/right side, random factor darkening
    '''
    out_img = img.astype(float)
    h, w, d = out_img.shape
    mid_h = np.random.randint(0, w)     # mid point horizontal
    mid_v = np.random.randint(0, h)     # mid point horizontal
    fac = np.random.uniform(0.5, 0.9)   # brightness factor
    hf = np.random.rand(0,2)            # 50% chance > 1
    if hf > 1:
        out_img[0:mid_v, :, 0] *= fac 
    else:
        out_img[mid_v:h, :, 0] *= fac
    return out_img

def horizon_shift(img):
    '''
    Random vertical translation - inspired by Jeremy's implementation
    '''
    out_img = img.astype(float)
    h, w, d = out_img.shape
    horizon = h*0.4
    v_shift = np.random.randint(-h/8, h/8)
    n_pts = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    m_pts = np.float32([[0, horizon+v_shift], [w, horizon+v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(n_pts, m_pts)
    out_img = cv2.warpPerspective(out_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out_img

def init_data():
    '''
    Method to import data from chosen locations, check if the car is moving
    '''
    imgs = []
    angs = []
    for j in range(2):
        if not DATA_CHECK[j]:
            print('not using ', str(CSV_PATH[j]))
            continue
        with open(CSV_PATH[j], newline='') as f:
            driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
        for row in driving_data[1:]:
            # skip if ~0 speed - not actually driving
            if float(row[6]) < 0.1:
                continue
            imgs.append(IMG_PATH[j] + row[0]) # Center
            angs.append(float(row[3]))
            imgs.append(IMG_PATH[j] + row[1]) # Left
            angs.append(float(row[3]) + 0.25)
            imgs.append(IMG_PATH[j] + row[2]) # Right
            angs.append(float(row[3]) - 0.25)
    return (np.array(imgs), np.array(angs))

def data_pipe(images, angles, batch_size=128, distort=True):
    '''
    Method to load, process, and distort images
    flips images with turning angle magnitudes of greater than 0.3
    to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    images, angles = shuffle(images, angles)
    data, labels = ([], [])
    while True:
        for ix in range(len(angles)):
            angle = angles[ix]
            img = preprocess_image(cv2.imread(images[ix]))
            if distort:
                img = random_distort(img) 
            data.append(img)
            labels.append(angle)
            if len(data) == batch_size:
                yield (np.array(data), np.array(labels))
                data, labels = ([], [])
                images, angles = shuffle(images, angles) 
            if abs(angle) > 0.3:
                img = cv2.flip(img, 1)
                angle *= -1
                data.append(img)
                labels.append(angle)
                if len(data) == batch_size:
                    yield (np.array(data), np.array(labels))
                    data, labels = ([], [])
                    images, angles = shuffle(images, angles)

def data_pipe_for_vis(images, angles, batch_size=20, distort=True):
    '''
    method for loading, processing, and distorting images
    '''
    data, labels = ([], [])
    images, angles = shuffle(images, angles)
    for k in range(batch_size):
        img = cv2.imread(images[k])
        img = preprocess_image(img)
        if distort:
            img = random_distort(img)
        data.append(img)
        angle = angles[k]
        labels.append(angle)
    return (np.array(data), np.array(labels))

MY_DATA = True
UDACITY_DATA = True
DATA_CHECK = [MY_DATA, UDACITY_DATA]
IMG_PATH = ['', getcwd() + '/udacity_data/']
CSV_PATH = ['./my_data/driving_log.csv', './udacity_data/driving_log.csv']

def main():
    '''
    Main program
    '''
    images_raw = [] # Data array
    angles_raw = [] # Label array
    images_raw, angles_raw = init_data()
    print('Before Processing:', images_raw.shape, angles_raw.shape)
    images, angles = normalize_data(images_raw, angles_raw, do_plot=True)
    print('After Processing:', images.shape, angles.shape)
    X_vis, y_vis = data_pipe_for_vis(images, angles)
    visualize_dataset(X_vis, y_vis)
    X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.05, random_state=42)
    print('Training Set Size:', X_train.shape, y_train.shape)
    print('Test Set Size:', X_test.shape, y_test.shape)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(32, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(8, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    train_gen = data_pipe(X_train, y_train, distort=True, batch_size=64)
    val_gen = data_pipe(X_train, y_train, distort=False, batch_size=64)
    test_gen = data_pipe(X_test, y_test, distort=False, batch_size=64)
    chpk = ModelCheckpoint('model{epoch:02d}.h5')
    history_object = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040, nb_epoch=5, verbose=2, callbacks=[chpk])
    print('Test Loss:', model.evaluate_generator(test_gen, 128))
    print(model.summary())
    ### print the keys contained in the history object
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    n = 12 # number of samples for visualization
    X_out, y_out = data_pipe_for_vis(X_test[:n], y_test[:n], batch_size=n)
    y_pred = model.predict(X_out, n, verbose=2)
    visualize_dataset(X_out, y_out, y_pred)
    model.save_weights('./model.h5') # Save model data
    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)
    print('Fin')

if __name__ == "__main__":
    main()