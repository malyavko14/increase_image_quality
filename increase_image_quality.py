import shutil
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
import cv2
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.metrics import structural_similarity as ssim


def neural_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), activation='linear', kernel_initializer='glorot_uniform',
                     padding='same', use_bias=True))

    SRCNN.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


class NeuralNetworkTraining:
    def __init__(self, path_to_folder):
        self.__path_to_folder = path_to_folder

    def image_conversion(self, path_to_file):
        image = cv2.imread(self.__path_to_folder + '/' + path_to_file)
        height, width, _ = image.shape
        new_height = int(height / 4)
        new_width = int(width / 4)

        input_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        input_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_CUBIC)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        X = np.zeros((1, input_image.shape[0], input_image.shape[1], 1), dtype=float)
        X[0, :, :, 0] = input_image[:, :, 0].astype(float) / 255
        Y = np.zeros((1, output_image.shape[0], output_image.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = output_image[:, :, 0].astype(float) / 255
        return X, Y

    def neural_network_training(self):
        for image in os.listdir(self.__path_to_folder):
            print(image)
            X, Y = self.image_conversion(image)

            SRCNN = neural_model()
            #SRCNN.load_weights('SRCNN_weights.hdf5')
            weights_file = "SRCNN_weights.hdf5"
            model = ModelCheckpoint(weights_file, monitor='mean_squared_error', mode='max', verbose=1)
            early_stop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001,
                                       patience=350, verbose=1, mode='auto')
            checkpoint = [model, early_stop]
            SRCNN.fit(x=X, y=Y, batch_size=32, epochs=1000, callbacks=[checkpoint])
            folder = "C:\\Users\\vivo\PycharmProjects\increase_image_quality\\trained_images"
            shutil.move('input\\' + image, folder)


class TestingNetwork:
    def __init__(self, file_path):
        self.__file_path = file_path
        self.__input_image = cv2.imread(self.__file_path)
        height, width, _ = self.__input_image.shape
        new_height = int(height / 4)
        new_width = int(width / 4)

        image = cv2.resize(self.__input_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        self.__cubic = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        self.__Y = np.zeros((1, image.shape[0], image.shape[1], 1), dtype=float)
        self.__Y[0, :, :, 0] = image[:, :, 0].astype(float) / 255
        self.__srcnn = neural_model()
        self.__srcnn.load_weights('SRCNN_weights.hdf5')
        self.__output = self.__srcnn.predict(self.__Y, batch_size=1)
        self.__output *= 255
        self.__output[self.__output[:] > 255] = 255
        self.__output[self.__output[:] < 0] = 0
        self.__output = self.__output.astype(np.uint8)

        image[:, :, 0] = self.__output[0, :, :, 0]
        self.__output_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

    def check_psnr(self):
        print('PSNR')
        cubic_PSNR = cv2.PSNR(self.__input_image, self.__cubic)
        SRCNN_PSNR = cv2.PSNR(self.__input_image, self.__output_image)
        print('Cubic: ', cubic_PSNR)
        print('SRCNN: ', SRCNN_PSNR)
        print('RSCNN better' if cubic_PSNR < SRCNN_PSNR else "Cubic better")

    def check_ssim(self):
        print('SSIM')
        cubic_SSIM = ssim(self.__input_image, self.__cubic, multichannel=True)
        SRCNN_SSIM = ssim(self.__input_image, self.__output_image, multichannel=True)
        print('Cubic: ', cubic_SSIM)
        print('SRCNN: ', SRCNN_SSIM)
        print('RSCNN better' if cubic_SSIM < SRCNN_SSIM else "Cubic better")

class SRCNNIncreaseImageQuality:
    def __init__(self, path_to_file, coefficient):
        self.__path_to_file = path_to_file
        self.image = cv2.imread(self.__path_to_file)
        self.coefficient = coefficient

    def image_conversion(self):
        height, width, _ = self.image.shape
        new_height = int(height * self.coefficient)
        new_width = int(width * self.coefficient)

        image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        Y = np.zeros((1, image.shape[0], image.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = image[:, :, 0].astype(float) / 255
        return Y, image

    def increase_image_quality(self):
        Y, image = self.image_conversion()
        srcnn = neural_model()
        srcnn.load_weights('SRCNN_weights.hdf5')
        output = srcnn.predict(Y, batch_size=1)
        output *= 255
        output[output[:] > 255] = 255
        output[output[:] < 0] = 0
        output = output.astype(np.uint8)

        image[:, :, 0] = output[0, :, :, 0]
        output_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        tmp = self.__path_to_file.split('/')[-1]
        new_name = f'{tmp.split(".")[0]}_improved_quality.{tmp.split(".")[-1]}'
        cv2.imwrite(f'result/{new_name}', output_image)


# SRCNNIncreaseImageQuality('input/f33cc2f847cd.jpg', 2).increase_image_quality()
#NeuralNetworkTraining('test').neural_network_training()
TestingNetwork('test/f33cc2f847cd.jpg').check_psnr()
TestingNetwork('test/f33cc2f847cd.jpg').check_ssim()
