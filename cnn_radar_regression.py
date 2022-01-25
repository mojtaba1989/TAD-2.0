import numpy as np
import os
import sys
import python_data_fusion as pdf
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from mobilenet_v3_block import BottleNeck, h_swish
import cv2
import copy

class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck10 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck11 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5)

        self.conv2 = tf.keras.layers.Conv2D(filters=576,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = h_swish(x)
        x = self.avgpool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return x

class pointCloudCNN(tf.keras.Model):
    def __init__(self, FILE_NAME, model, augmentation=False, pointPower=200):
        super(pointCloudCNN, self).__init__()
        ## DIrectories
        self.FILE_NAME          = FILE_NAME
        self.PATH_TO_CSV        = {}
        self.PATH_TO_RESULTS    = {}
        self.PATH_TO_CKPT       = {}
        self.PATH_TO_PB         = {}
        self.PATH_TO_FOLDER     = {}
        self.PATH_TO_H5         = {}
        ## Active Variables
        self.model          = model
        self.newList        = []
        self.counter        = 0
        self.dataNum        = 0
        self.result         = {}
        self.Lables         = []
        self.timeStamp      = 0
        self.refHitchAngle  = 0
        self.radarImages    = []
        self.CNN            = []
        self.trainAttrX     = []
        self.testAttrX      = []
        self.trainImagesX   = []
        self.testImagesX    = []
        self.trainY         = []
        self.testY          = []
        self.validationX    = []
        self.validationY    = []
        self.opt            = []
        self.callBack       = []
        self.batchSize      = 16
        self.ckpt           = []
        self.manager        = []
        ## Passive Varibales
        self.center         = [-0.03991714, 0.35]
        ## Handles
        self.setDirectory()
        self.constants      = self.constantsMakeObject()
        self.augmentation   = augmentation
        self.virtualObj     = []
        self.pointPower     = pointPower
        self.virtualObjBuild()
        ## External Data Set
        self.dataID         = 0
        self.externalData   = []
        self.meanSNR         = []
        ## Print Colors
        self.CRED           = '\033[91m'
        self.CEND           = '\033[0m'

    class constantsMakeObject:
        def __init__(self):
            self.rLim       = (1, 3)
            self.tLim       = (-85, 85)
            self.resolution = (75, 171, 3)
            self.r_edges = np.linspace(self.rLim[0], self.rLim[1], self.resolution[0])
            self.t_edges = np.linspace(self.tLim[0], self.tLim[1], self.resolution[1])

    def setDirectory(self):
        CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(CWD_PATH, 'pointCloudCNN_Models', self.model)
        if not os.path.isdir(self.PATH_TO_CKPT):
            os.makedirs(self.PATH_TO_CKPT)
        self.PATH_TO_PB = os.path.join(self.PATH_TO_CKPT, 'frozen_graph.pb')
        self.PATH_TO_H5 = os.path.join(self.PATH_TO_CKPT, 'model.h5')

    def parsDir(self, file_name):
        CWD_PATH = os.getcwd()
        self.PATH_TO_CSV = os.path.join(CWD_PATH, file_name, file_name + '.csv')
        self.PATH_TO_RESULTS = os.path.join(CWD_PATH, file_name, file_name + '-RESULT.csv')
        self.PATH_TO_FOLDER = os.path.join(CWD_PATH, file_name)
        dummy = pd.read_csv(self.PATH_TO_CSV)
        self.dataNum = len(dummy)

    class makeObj:
        def __init__(self, x, y, peakVal, center, sensor):
            self.x = x - center[0]
            self.y = y - center[1]
            self.peakVal = peakVal
            self.range = np.sqrt(self.x ** 2 + self.y ** 2)
            self.azimuth = np.degrees(np.arctan(self.x / self.y))
            self.sensor = sensor

    def virtualObjBuild(self):
        if self.augmentation:
            self.virtualObj = []
            point_num = np.random.randint(1, 5)
            for _ in range(point_num):
                self.virtualObj.append(self.makeObj(np.random.uniform(-.25, .25),
                                                    np.random.uniform(1.25, 1.75),
                                                    self.pointPower,
                                                    self.center, "virtual"))

    def dataParser(self):
        index, self.timeStamp, self.refHitchAngle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d \
            = pdf.readCSV(self.PATH_TO_CSV, self.counter + 1)
        p_p = pdf.tm_f(p_p, .16, .84, .05, 20, 7, 'p')
        p_d = pdf.tm_f(p_d, .16, .84, .05, 20, 7, 'd')
        self.normalizedHitchAngle = self.refHitchAngle/180 + 0.5
        self.newList = []
        for i in range(len(x_p)):
            self.newList.append(self.makeObj(p_p[0, i], p_p[1, i], peakVal_p[i], self.center, "passenger"))
            self.meanSNR.append(peakVal_p[i])
        for i in range(len(x_d)):
            self.newList.append(self.makeObj(p_d[0, i], p_d[1, i], peakVal_d[i], self.center, "driver"))
            self.meanSNR.append(peakVal_d[i])
        if self.augmentation:
            dummy = copy.deepcopy(self.virtualObj)
            for obj in dummy:
                obj.azimuth += self.refHitchAngle + np.random.normal(0, .25)
                obj.range += np.random.normal(0, .01)
                obj.peakVal += np.random.randint(0, 100)
                self.newList.append(obj)

    def occupancyGridGenerator(self, train=True):
        radar_image = np.zeros(self.constants.resolution[0:2])
        for obj in self.newList:
            if self.constants.rLim[0] <= obj.range <= self.constants.rLim[1] \
                    and self.constants.tLim[0] <= obj.azimuth <= self.constants.tLim[1]:
                indx = np.digitize(obj.azimuth, self.constants.t_edges)
                indy = np.digitize(obj.range, self.constants.r_edges)
                radar_image[indy, indx] =+ obj.peakVal
        if np.max(np.max(radar_image)) > 0:
            radar_image = radar_image / np.max(np.max(radar_image))
        uint_img = np.array(radar_image * 255).astype('uint8')
        grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

        if train:
            self.radarImages.append(grayImage)
        else:
            img_name = 'img%d.jpeg' % self.counter
            if not os.path.isdir(os.path.join(self.PATH_TO_FOLDER, 'radar_images')):
                os.makedirs(os.path.join(self.PATH_TO_FOLDER, 'radar_images'))
            cv2.imwrite(os.path.join(self.PATH_TO_FOLDER, 'radar_images', img_name), grayImage)
            return grayImage

    def dataBuild(self, split=False, fine_tuning=False):
        for fileName in self.FILE_NAME:
            self.parsDir(fileName)
            self.counter = 0
            for i in range(self.dataNum):
                self.dataParser()
                self.Lables.append(self.refHitchAngle)
                self.occupancyGridGenerator()
                self.counter += 1
        if split and not fine_tuning:
            self.Lables = pd.DataFrame(data=np.array(self.Lables), columns=['Vernier'])
            (self.trainAttrX, self.testAttrX, self.trainImagesX, self.testImagesX) =\
                train_test_split(self.Lables, np.array(self.radarImages), test_size=0.25, random_state=42)
            self.trainY = self.trainAttrX["Vernier"]
            self.testY = self.testAttrX["Vernier"]
        elif split:
            self.dataPrep()
            self.Lables = pd.DataFrame(data=np.array(self.Lables), columns=['Vernier'])
            (self.trainAttrX, self.testAttrX, self.trainImagesX, self.testImagesX) = \
                train_test_split(self.Lables, np.array(self.radarImages), test_size=0.2, random_state=42)
            self.trainY = self.trainAttrX["Vernier"]
            self.testY = self.testAttrX["Vernier"]
        else:
            self.Lables = pd.DataFrame(data=np.array(self.Lables), columns=['Vernier'])

    def dataPrep(self, d=1, N=2):
        self.dataNum = len(self.Lables)
        dummyLable = np.array(self.Lables)
        self.radarImages = [self.radarImages[i] for i in dummyLable.argsort()]
        self.Lables = [self.Lables[i] for i in dummyLable.argsort()]
        indicator = np.ones(self.dataNum, dtype=bool)
        n=0
        dummy = self.Lables[0]
        for i in range(1, self.dataNum):
            if self.Lables[i]-dummy < d:
                n = n + 1
                if n > N: indicator[i] = False
            else:
                dummy = self.Lables[i]
                n = 0
        self.radarImages    = [self.radarImages[i] for i in range(self.dataNum) if indicator[i]]
        self.Lables         = [self.Lables[i] for i in range(self.dataNum) if indicator[i]]
        print('[INFO]: DATA SET IS CLEANED--original:{}---->cleaned{}'.format(self.dataNum, len(self.Lables)))

    def modelParSet(self):
        self.ckpt           = tf.train.Checkpoint(step=tf.Variable(1), opt=self.opt, model=self.CNN)
        self.manager        = tf.train.CheckpointManager(self.ckpt, self.PATH_TO_CKPT, max_to_keep=3)

    def cnnModel(self):
        from tensorflow import keras
        from tensorflow.keras import layers
        from keras.utils.generic_utils import get_custom_objects

        def custom_activation(x):
            return (x - .5) / 180
        get_custom_objects().update({'custom_activation': layers.Activation(custom_activation)})
        def residual_module(layer_in, n_filters):
            merge_input = layer_in
            if layer_in.shape[-1] != n_filters:
                merge_input = layers.Conv2D(n_filters, (1, 1), padding='same', activation='relu',
                                     kernel_initializer='he_normal')(
                    layer_in)
            conv1 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(
                layer_in)
            conv2 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(
                conv1)
            layer_out = layers.add([conv2, merge_input])
            layer_out = layers.Activation('relu')(layer_out)
            return layer_out
        def myInception(layer_in, f1, f2, f3, f4, identity=False):
            conv1_1 = layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(layer_in)
            conv1_2 = layers.Conv2D(filters=f2, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
            conv1_3 = layers.Conv2D(filters=f3, kernel_size=(3, 3), padding='same', activation='relu')(conv1_2)
            pool    = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer_in)
            pool   = layers.Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(pool)
            layer_out = layers.concatenate([conv1_1, conv1_2, conv1_3, pool])
            if identity:
                merge_in = layer_in
                if layer_in.shape[-1] != f1+f2+f3+f4:
                    merge_input = layers.Conv2D(f1+f2+f3+f4, (1, 1), padding='same', activation='relu')(layer_in)
                layer_out = layers.add([layer_out, merge_input])
                layer_out = layers.Activation('relu')(layer_out)
            return layer_out
        def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
            conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
            conv3 = layers.Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
            conv3 = layers.Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
            conv5 = layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
            conv5 = layers.Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
            pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
            pool = layers.Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
            layer_out = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
            return layer_out

        if self.model == 'NASNetMobile':
            from tensorflow.keras.applications.nasnet import NASNetMobile
            temp = NASNetMobile(input_shape=self.constants.resolution,
                               include_top=False,
                               weights=None)
            x = layers.GlobalAveragePooling2D()(temp.output)
            x = layers.Flatten()(x)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=temp.input, outputs=x)
        elif self.model == 'MojNet':
            self.CNN = keras.Sequential([
                layers.Conv2D(input_shape=self.constants.resolution, filters=16, kernel_size=(3, 3), padding="same",
                              activation="relu"),
                layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.GlobalMaxPool2D(),
                layers.Dense(100, activation='relu'),
                layers.Dense(20, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
        elif self.model == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3
            temp = InceptionV3(input_shape=self.constants.resolution,
                           include_top=False,
                           weights=None)
            x = layers.DepthwiseConv2D(kernel_size=(1, 4))(temp.output)
            x = layers.Flatten()(x)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=temp.input, outputs=x)
        elif self.model == 'InceptionMoj':
            input = layers.Input(shape=self.constants.resolution,
                                 tensor=tf.compat.v1.placeholder('float32', shape=(1, )+self.constants.resolution))
            x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")(input)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
            x = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = myInception(x, 64, 128, 32, 32)
            x = myInception(x, 128, 192, 96, 64)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = myInception(x, 192, 208, 48, 64)
            x = myInception(x, 160, 224, 64, 64)
            x = myInception(x, 128, 256, 64, 64)
            x = myInception(x, 112, 288, 64, 64)
            x = myInception(x, 256, 320, 128, 128)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = layers.Flatten()(x)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=input, outputs=x)
        elif self.model == 'InceptionMoj2':
            input = layers.Input(shape=self.constants.resolution)
            x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation="relu")(input)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
            x = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = myInception(x, 64, 128, 32, 32)
            x = myInception(x, 128, 192, 96, 64)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = myInception(x, 192, 208, 48, 64)
            x = myInception(x, 160, 224, 64, 64)
            x = myInception(x, 128, 256, 64, 64)
            x = myInception(x, 112, 288, 64, 64)
            x = myInception(x, 256, 320, 128, 128)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
            x = layers.DepthwiseConv2D(kernel_size=(1, 4))(x)
            x = layers.Flatten()(x)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear", name='estimated_angle')(x)
            # x = layers.Activation(custom_activation, name='SpecialActivation')(x)
            self.CNN = keras.models.Model(inputs=input, outputs=x)
        elif self.model == 'MobileNet2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            temp = MobileNetV2(input_shape=self.constants.resolution,
                                include_top=False,
                                weights=None)
            x = layers.DepthwiseConv2D(kernel_size=(3, 6), activation='relu')(temp.output)
            x = layers.Flatten()(x)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=temp.input, outputs=x)
        elif self.model == 'MobileNet22':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            temp = MobileNetV2(input_shape=self.constants.resolution,
                                include_top=False,
                                weights=None)
            x = layers.GlobalMaxPool2D()(temp.output)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=temp.input, outputs=x)
        elif self.model == 'MobileNet3':
            self.CNN = MobileNetV3Small()
            self.CNN.build(input_shape=(None, 75, 171, 3))
        elif self.model == 'VGG':
            from tensorflow.keras.applications.vgg16 import VGG16 as PTModel
            temp = PTModel(input_shape=self.constants.resolution,
                                include_top = False,
                                weights=None)
            x = layers.GlobalAveragePooling2D()(temp.output)
            x = layers.Dense(200)(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Dense(20, activation="relu")(x)
            x = layers.Dense(10, activation="relu")(x)
            x = layers.Dense(1, activation="linear")(x)
            self.CNN = keras.models.Model(inputs=temp.input, outputs=x)

        self.opt = Adam(lr=1e-4, decay=1e-3)
        self.CNN.compile(loss="MSE", optimizer=self.opt)

    def cnnFit(self, epochs=10, save=False):
        checkpoint_prefix = os.path.join(self.PATH_TO_CKPT, "ckpt")
        try:
            self.ckpt.restore(self.manager.latest_checkpoint)
        except:
            self.modelParSet()
        try:
            self.CNN.load_weights(self.manager.latest_checkpoint)
        except:
            pass
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        while not save:
            self.CNN.fit(x=self.trainImagesX, y=self.trainY,
                         validation_data=(self.testImagesX, self.testY),
                         epochs=epochs,
                         batch_size=self.batchSize)

            self.ckpt.step.assign_add(1)
            save_path = self.manager.save()
            tf.keras.Model.save_weights(self.CNN, filepath=save_path)
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            preds = self.CNN.predict(self.testImagesX)
            error = preds.flatten() - radarCNN.testY
            print("[INFO] RME: {:.4f}".format(np.sqrt(np.mean(error ** 2))))
            self.publishEval()

    def train(self):
        self.dataBuild(split=True)
        self.setDirectory()
        self.cnnModel()
        self.CNN.summary()
        self.cnnFit(save=False)

    def save(self):
        self.dataBuild(split=True)
        self.setDirectory()
        self.cnnModel()
        self.cnnFit(save=True)
        self.CNN.save(self.PATH_TO_H5)
        self.CNN.summary()
        self.publishEval()

    def loadPb(self):
        with tf.io.gfile.GFile(self.PATH_TO_PB, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def getFlops(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from tensorflow.python.keras import backend as K
        run_meta = tf.RunMetadata()
        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            with tf.device('/cpu:0'):
                base_model = self.CNN
                opts = tf.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

                opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
                params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

    def test(self, display=False):
        self.setDirectory()
        self.CNN = tf.keras.models.load_model(self.PATH_TO_H5)
        for fileName in self.FILE_NAME:
            self.parsDir(fileName)
            self.counter = 0
            temp = pd.DataFrame(np.zeros([self.dataNum, 2]), columns=["Vernier", "Radar"])
            for i in range(self.dataNum):
                self.dataParser()
                radarImage = self.occupancyGridGenerator(train=False)
                if display:
                    cv2.imshow('Occupancy Grid Map', cv2.resize(radarImage, (800, 600)))
                    cv2.waitKey(1)
                radarImage = np.expand_dims(radarImage, axis=0)
                prediction = self.CNN.predict(radarImage)
                temp.loc[self.counter, "Vernier"] = np.array(self.refHitchAngle)
                temp.loc[self.counter, "Radar"] = prediction
                self.counter += 1
            temp.to_csv(self.PATH_TO_RESULTS)
        self.publishEval()

    class externalDataObj:
        def __init__(self):
            self.resultTable    = {}
            self.imageStack     = []
            self.dataNum        = []
            self.RMS            = 0
            self.AR             = 0
            self.AT             = 0
            self.R2             = 0
            self.info           = {}
            self.id             = 0
            self.X              = []
            self.Y              = []
            self.meanSNR        = 0
            self.refPower       = 0

    def externalDataAdd(self, fileName, info):
        self.parsDir(fileName)
        self.counter = 0
        dummyObj = []
        dummyObj = self.externalDataObj()
        dummyObj.dataNum = self.dataNum
        dummyObj.info = info
        dummyObj.id = self.dataID
        dummyObj.resultTable = pd.DataFrame(np.zeros([self.dataNum, 1]), columns=["Vernier"])
        self.meanSNR = []
        dummyObj.refPower = self.pointPower
        for i in range(self.dataNum):
            self.dataParser()
            dummyObj.imageStack.append(self.occupancyGridGenerator(train=False))
            dummyObj.resultTable.loc[self.counter, "Vernier"] = np.array(self.refHitchAngle)
            self.counter += 1
        dummyObj.X = np.array(dummyObj.imageStack)
        dummyObj.Y = dummyObj.resultTable["Vernier"]
        dummyObj.meanSNR = np.mean(self.meanSNR)
        self.dataID += 1
        self.externalData.append(dummyObj)

    def toDict(self, obj):
        if not self.augmentation:
            return {
                "id" : obj.id,
                "NUM" : obj.dataNum,
                "INFO": obj.info,
                "RMS": obj.RMS,
                "R2":  obj.R2,
                "AR": obj.AR,
                "Average Time": obj.AT
            }
        else:
            return {
                "id" : obj.id,
                "NUM" : obj.dataNum,
                "INFO": obj.info,
                "RMS": obj.RMS,
                "R2":  obj.R2,
                "AR": obj.AR,
                "PointPower": obj.refPower,
                "MaxSNR": obj.meanSNR
            }

    def publishEval(self):
        from sklearn.metrics import r2_score as r2_score
        if self.externalData:
            for obj in self.externalData:
                t_start = time.time()
                preds = self.CNN.predict(obj.X)
                t_end = time.time()
                error = preds.flatten() - obj.Y
                obj.RMS = np.sqrt(np.mean(error ** 2))
                obj.AR = np.mean(np.abs(error) <= np.abs(obj.Y)/10 + 1)
                obj.AT = (t_end-t_start) / obj.dataNum
                obj.R2 = r2_score(obj.Y, preds.flatten())
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(pd.DataFrame([self.toDict(obj) for obj in self.externalData]))
            # print('[INFO] Model has {} layers'.format(len(self.CNN.layers)))

    def fineTuning(self, epochs=20, save=False, random=.05):
        self.dataBuild(split=True, fine_tuning=True)
        self.setDirectory()
        self.cnnModel()
        self.cnnFit(save=True)
        self.publishEval()
        tstart = time.time()
        index = 0
        while True:
            if not 'depthwise' in self.CNN.layers[index].name:
                self.CNN.layers[index]._trainable = False
                index += 1
            else:
                break
        for ix, layer in enumerate(self.CNN.layers):
            if 'dense' in layer.name:
                if hasattr(self.CNN.layers[ix], 'kernel_initializer') and \
                        hasattr(self.CNN.layers[ix], 'bias_initializer'):
                    weight_initializer = self.CNN.layers[ix].kernel_initializer
                    bias_initializer = self.CNN.layers[ix].bias_initializer

                    old_weights, old_biases = self.CNN.layers[ix].get_weights()

                    self.CNN.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=old_biases.shape)])
        self.CNN.compile(loss="MSE", optimizer=Adam(lr=1e-3, decay=1e-3))
        self.CNN.fit(x=self.trainImagesX, y=self.trainY,
                     validation_data=(self.testImagesX, self.testY),
                     epochs=150,
                     batch_size=self.batchSize)
        print('[INFO] Feature Extraction Training Is Disabled')
        for ix, layer in enumerate(self.CNN.layers):
            if np.random.uniform() > (1 - random):
                try:
                    self.CNN.layers[ix]._trainable = True
                except:
                    pass

        self.CNN.compile(loss="MSE", optimizer=self.opt)
        print('[INFO]: {}% Of Conv2D Layers Will Be Tuned'.format(int(random*100)))
        self.CNN.fit(x=self.trainImagesX, y=self.trainY,
                     validation_data=(self.testImagesX, self.testY),
                     epochs=50,
                     batch_size=self.batchSize)
        tend = time.time()
        if save:
            self.ckpt.step.assign_add(1)
            save_path = self.manager.save()
            tf.keras.Model.save_weights(self.CNN, filepath=save_path)
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        self.publishEval()
        print('[INFO] Fine-Tuning Overal Time:{}'.format(tend-tstart))





if not __name__ == '__main__':
    # FILE_NAME = {'10-Aug-2020-16-29', '10-Aug-2020-16-38',
    #              '10-Aug-2020-17-47', '28-Sep-2020-14-18',
    #              '02-Nov-2020-13-36', '09-Nov-2020-13-36'}
    # FILE_NAME = {'10-Aug-2020-16-29', '10-Aug-2020-16-38',
    #              '02-Nov-2020-13-36', '09-Nov-2020-13-36'}
    FILE_NAME = {}
    radarCNN = pointCloudCNN(FILE_NAME, 'InceptionMoj2', augmentation=True)
    radarCNN.externalDataAdd('10-Aug-2020-16-50', 'W Corner')
    radarCNN.externalDataAdd('28-Sep-2020-15-50', 'Enclosed')
    radarCNN.externalDataAdd('09-Nov-2020-14-24', 'FlatBed')
    # radarCNN.train()
    # radarCNN.save()
    radarCNN.test(display=True)
    # radarCNN.fineTuning()

if __name__ == '__main__':
    radarCNN = pointCloudCNN({}, 'InceptionMoj2', augmentation=True, pointPower=0)
    for i in range(100):
        radarCNN.pointPower = np.random.randint(1, 480)
        radarCNN.virtualObjBuild()
        radarCNN.externalDataAdd('09-Nov-2020-14-24', 'FlatBed')
        print(i, radarCNN.pointPower)

    radarCNN.test()







