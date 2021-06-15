import shutil

from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.color import rgb2lab, lab2rgb

import cv2
import numpy as np
import os

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt


def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), activation='linear', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True))

    SRCNN.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


class NeuralNetworkTraining:
    def __init__(self, path_to_folder):
        self.__path_to_folder = path_to_folder

    def modcrop(self, img, scale):
        tmpsz = img.shape
        sz = tmpsz[0:2]
        sz = sz - np.mod(sz, scale)
        img = img[0:sz[0], 1:sz[1]]
        return img

    def shave(self, image, border):
        img = image[border: -border, border: -border]
        return img

    def image_conversion(self, path_to_file):
        output_image = cv2.imread(self.__path_to_folder + '/' + path_to_file)
        height, width, _ = output_image.shape
        new_height = int(height / 4)
        new_width = int(width / 4)

        input_image = cv2.resize(output_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        input_image = cv2.resize(output_image, (width, height), interpolation=cv2.INTER_CUBIC)
        input_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCrCb)
        '''
        degraded = input_image
        degraded = self.modcrop(degraded, 3)
        temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
        Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
        pre = srcnn.predict(Y, batch_size=1)

        # post-process output
        pre *= 255
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)

        # copy Y channel back to image and convert to BGR
        temp = self.shave(temp, 6)
        temp[:, :, 0] = pre[0, :, :, 0]
        output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

        # remove border from reference and degraged image
        degraded = self.shave(degraded.astype(np.uint8), 6)
        '''

        X = np.zeros((1, output_image.shape[0], output_image.shape[1], 1), dtype=float)
        X[0, :, :, 0] = output_image[:, :, 0].astype(float) / 255
        Y = np.zeros((1, input_image.shape[0], input_image.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = input_image[:, :, 0].astype(float) / 255

        return X, Y




    def neural_network_training(self):
        for image in os.listdir(self.__path_to_folder):
            print(image)
            X, Y = self.image_conversion(image)

            SRCNN = model()
            # SRCNN.load_weights('SRCNN_weights.hdf5')
            weights_file = "SRCNN_weights.hdf5"
            checkpoint = ModelCheckpoint(weights_file, monitor='mean_squared_error', mode='max', save_best_only=True, verbose=1)
            SRCNN.fit(x=X, y=Y, batch_size=1, epochs=100, callbacks=[checkpoint])
            #shutil.move(image, "trained_images")


class SRCNNIncreaseImageQuality:
    def __init__(self, path_to_file, coefficient):
        self.__path_to_file = path_to_file
        self.coefficient = coefficient

    def image_conversion(self):
        image = cv2.imread(self.__path_to_file)
        height, width, _ = image.shape
        new_height = int(height * self.coefficient)
        new_width = int(width * self.coefficient)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        Y = np.zeros((1, image.shape[0], image.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = image[:, :, 0].astype(float) / 255
        return Y, image

    def increase_image_quality(self):
        Y, image = self.image_conversion()
        srcnn = model()
        srcnn.load_weights('SRCNN_weights.hdf5')
        output = srcnn.predict(Y, batch_size=1)
        output *= 255
        output[output[:] > 255] = 255
        output[output[:] < 0] = 0
        output = output.astype(np.uint8)

        #image = image[6: -6, 6: -6]
        image[:, :, 0] = output[0, :, :, 0]
        output_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        tmp = self.__path_to_file.split('/')[-1]
        cv2.imwrite(f'{tmp.split(".")[0]}_improved_quality.{tmp.split(".")[-1]}', output_image)

        #fig, axs = plt.subplots(1, 2, figsize=(40, 16))
        #axs[0].imshow(cv2.cvtColor(cv2.imread(self.__path_to_file), cv2.COLOR_BGR2RGB))
        #axs[0].set_title('Original')
        #axs[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        #axs[1].set_title('Degraded')
        #for ax in axs:
        #    ax.set_xticks([])
        #    ax.set_yticks([])

        #fig.savefig('output'.format(tmp.split(".")[-1]))
        #plt.close()


#SRCNNIncreaseImageQuality('input/ex.jpg', 2).increase_image_quality()
NeuralNetworkTraining('input').neural_network_training()
