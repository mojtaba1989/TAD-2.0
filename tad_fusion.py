import numpy as np
import os
import sys
import python_data_fusion as pdf
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pandas as pd
import time
from operator import attrgetter
from scipy.signal import savgol_filter
from scipy import optimize
from scipy import stats
import copy

import cv2
import tensorflow as tf
from object_detection.utils import label_map_util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")

def imgFilter(img, u_lim=[0, 1], v_lim=[0, 1]):
    u = (np.array(u_lim)*img.shape[1]).astype(int)
    v = (np.array(v_lim)*img.shape[0]).astype(int)
    mask = np.zeros(img.shape[0:2], np.uint8)
    mask[v[0]:v[1], u[0]:u[1]] = 255
    mask_inv = cv2.bitwise_not(mask)
    image_masked = cv2.bitwise_and(img, img, mask=mask).copy()
    solid_image = np.zeros(img.shape, np.uint8)
    solid_image[:] = img.mean(axis=0).mean(axis=0).astype(int)
    background = cv2.bitwise_and(solid_image, solid_image, mask=mask_inv)
    filtered_image =  cv2.add(image_masked, background)

    return filtered_image

def centerCalc(x, y, initial_center=(0, 0)):
    def calc_R(xc, yc):
        return np.sqrt((y - yc) ** 2 + (x - xc) ** 2)
    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    def Df_2b(c):
        xc, yc = c
        df2b_dc = np.empty((2, len(x)))
        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x) / Ri
        df2b_dc[1] = (yc - y) / Ri
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]
        return df2b_dc
    center, ier = optimize.leastsq(f, initial_center, Dfun=Df_2b, col_deriv=True, maxfev=400)
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    residul = np.sqrt(np.mean((Ri - R) ** 2))
    return center.reshape(2, 1), R, residul

class pointCloudProcessing:
        def __init__(self, FILE_NAME, hitchAngle):
            self.trackingObjects = []
            self.newList = []
            self.waitingList = []
            self.topDeletedList = []
            self.constellation = []
            self.center = [-0.03991714, 0.35]
            self.constantObj = self.constant()
            self.kalmanObj = self.kalmanMakeObj(hitchAngle)
            self.counter = 0
            self.PATH_TO_CSV = {}
            self.FILE_NAME = FILE_NAME
            self.dataNum = 0
            self.result = {}
            self.PATH_TO_RESULTS = {}
            self.timeStamp = 0
            self.refHitchAngle = 0
            self.refAngleStack = []
            self.currentID = 0
            self.setDirectory()
            self.detectionFlag = False
            self.constellationMapIsReady = False
            self.maxConstellationPoints  = 10
            self.minConstellationPoints  = 5
            self.falseTrack = 0
            self.maxFalseTrack = 3

        class makeObj:
            def __init__(self, x, y, peakVal, center, score, hitchAngle, sensor, currentId):
                self.id = currentId
                self.x = x - center[0]
                self.y = y - center[1]
                self.peakVal = peakVal
                self.range = np.sqrt(self.x ** 2 + self.y ** 2)
                self.azimuth = np.degrees(np.arctan(self.x / self.y))
                self.score = score
                self.sensor = sensor
                self.state = "new"
                self.relativeAzimuth = self.azimuth - hitchAngle
                self.dAzimuth = 0
                self.dRange = 0
                self.age = 0
                self.kalman = self.kalmanMakeObjtracker(self.azimuth, self.range)

            class kalmanMakeObjtracker:
                def __init__(self, azimuth, range):
                    self.X = np.array([[azimuth], [0], [range]])
                    self.A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
                    self.P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, .04 ** 2]])
                    self.K = np.eye(3)
                    self.Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, .04 ** 2]])
                    self.R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, .04 ** 2]])

        class constant:
            def __init__(self):
                ## filtering constants
                self.minRange = 1
                self.maxRange = 3
                self.azimuthWindow = 40
                self.powerLB = 10
                self.powerUB = 500
                self.minpower = self.powerLB
                ## tracking constants
                self.rangeWeight = 100
                self.azimuthWeight = 1
                self.peakValWeight = 0
                self.waitingListWeight = 1
                self.topDeletedListWeight = 1
                self.directionmatchprize = .6
                self.maxDistance = np.sqrt(5 ** 2 + 8 ** 2)
                ## scoring constants
                self.dRangeWeight = 4
                self.dAzimuthWeight = 1
                self.initialScore = 0.01

        class kalmanMakeObj:
            def __init__(self, hitchAngle):
                self.azimuth = float(hitchAngle)
                self.azimuthVelocity = 0
                self.confidence = 1

        def toDict(self, obj):
            return {
                "id": obj.id,
                "age": obj.age,
                # "x": obj.x,
                # "y": obj.y,
                "peakVal": obj.peakVal,
                "range": obj.range,
                "azimuth": obj.azimuth,
                "score": obj.score,
                # "sensor": obj.sensor,
                # "state": obj.state,
                # "relativeAzimuth": obj.relativeAzimuth,
                "dAzimuth": obj.dAzimuth
                # "dRange": obj.dRange
            }

        def setDirectory(self):
            CWD_PATH = os.getcwd()
            self.PATH_TO_CSV = os.path.join(CWD_PATH, self.FILE_NAME, self.FILE_NAME + '.csv')
            self.PATH_TO_RESULTS = os.path.join(CWD_PATH, self.FILE_NAME, 'RADAR-lined-' + self.FILE_NAME + '.csv')
            dummy = pd.read_csv(self.PATH_TO_CSV)
            self.dataNum = len(dummy)
            self.result = pd.DataFrame(np.zeros([self.dataNum, 1]), columns=["Vernier"])

        def dataParser(self, external_source):
            index, self.timeStamp, self.refHitchAngle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d \
                = pdf.readCSV(
                self.PATH_TO_CSV, self.counter + 1)
            p_p = pdf.tm_f(p_p, .16, .84, .05, 20, 7, 'p')
            p_d = pdf.tm_f(p_d, .16, .84, .05, 20, 7, 'd')
            self.newList = []
            self.refAngleStack.append(self.refHitchAngle)
            while self.refAngleStack.__len__() > 10:
                self.refAngleStack.pop(0)
            if not self.detectionFlag:
                if external_source == np.nan:
                    currentHitchAngle = self.kalmanObj.azimuth
                else:
                    currentHitchAngle = external_source
                for i in range(len(x_p)):
                    self.newList.append(self.makeObj(p_p[0, i], p_p[1, i], peakVal_p[i],
                                                     self.center, self.constantObj.initialScore, currentHitchAngle,
                                                     "passenger",
                                                     self.currentID))
                    self.currentID += 1
                    if not np.abs(self.newList[-1].azimuth - currentHitchAngle) <= self.constantObj.azimuthWindow or \
                            not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange) \
                            or not self.newList[-1].peakVal > self.constantObj.minpower:
                        self.newList.pop()
                        self.currentID -= 1
                for i in range(len(x_d)):
                    self.newList.append(self.makeObj(p_d[0, i], p_d[1, i], peakVal_d[i],
                                                     self.center, self.constantObj.initialScore, currentHitchAngle,
                                                     "driver",
                                                     self.currentID))
                    self.currentID += 1
                    if not np.abs(self.newList[-1].azimuth - currentHitchAngle) <= self.constantObj.azimuthWindow or \
                            not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange) \
                            or not self.newList[-1].peakVal > self.constantObj.minpower:
                        self.newList.pop()
                        self.currentID -= 1
            else:
                for i in range(len(x_p)):
                    self.newList.append(self.makeObj(p_p[0, i], p_p[1, i], peakVal_p[i],
                                                     self.center, self.constantObj.initialScore, 0,
                                                     "passenger",
                                                     self.currentID))
                    self.currentID += 1
                    if not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange) \
                            or not self.newList[-1].peakVal > self.constantObj.minpower:
                        self.newList.pop()
                        self.currentID -= 1
                for i in range(len(x_d)):
                    self.newList.append(self.makeObj(p_d[0, i], p_d[1, i], peakVal_d[i],
                                                     self.center, self.constantObj.initialScore, 0,
                                                     "driver",
                                                     self.currentID))
                    self.currentID += 1
                    if not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange) \
                            or not self.newList[-1].peakVal > self.constantObj.minpower:
                        self.newList.pop()
                        self.currentID -= 1
            if self.newList:
                self.constantObj.powerUB = np.min(
                    [self.constantObj.powerUB, np.median([obj.peakVal for obj in self.newList])])

        def clustering(self, list, measure='peakVal'):
            if list and list.__len__() >= 2:
                D = np.empty([list.__len__(), list.__len__()])
                for j in range(list.__len__()):
                    D[j, ] = self.distFun(list[j], list, directivity=False)
                for j in range(list.__len__()):
                    D[j, j] = np.max(D)
                if measure == 'peakVal':
                    while not D.size <= 2 and np.min(D) <= self.constantObj.maxDistance / 10:
                        matchIndex = np.unravel_index(D.argmin(), D.shape)
                        list.pop(matchIndex[1])
                        D = np.delete(D, matchIndex[1], axis=1)
                        D = np.delete(D, matchIndex[1], axis=0)
                elif measure == 'score':
                    while not D.size <= 2 and np.min(D) <= self.constantObj.maxDistance /5:
                        matchIndex = np.unravel_index(D.argmin(), D.shape)
                        if list[matchIndex[0]].score >= list[matchIndex[1]].score:
                            list.pop(matchIndex[1])
                            D = np.delete(D, matchIndex[1], axis=1)
                            D = np.delete(D, matchIndex[1], axis=0)
                        else:
                            list.pop(matchIndex[0])
                            D = np.delete(D, matchIndex[0], axis=1)
                            D = np.delete(D, matchIndex[0], axis=0)

        def distFun(self, obj, list, directivity=True):
            D = np.empty([3, list.__len__()])
            c = (self.constantObj.directionmatchprize + 1) / 2
            s = (-self.constantObj.directionmatchprize + 1) / 2
            for i in range(list.__len__()):
                D[0, i] = (obj.range - list[i].range) * self.constantObj.rangeWeight
                D[1, i] = (obj.azimuth - list[i].azimuth) * self.constantObj.azimuthWeight
                D[2, i] = (obj.peakVal - list[i].peakVal) * self.constantObj.peakValWeight
            if self.kalmanObj.azimuthVelocity >= 0:
                Dangle = c - s * np.cos(np.arctan2(D[0,], D[1,]))
            else:
                Dangle = c - s * np.cos(np.pi - np.arctan2(D[0,], D[1,]))
            if directivity:
                Dsum = np.sqrt(np.sum(np.abs(D) ** 2, axis=0)) * Dangle
            else:
                Dsum = np.sqrt(np.sum(np.abs(D) ** 2, axis=0))
            return Dsum

        def updateObject(self, obj_origin, obj_new, method='none'):
            obj_origin.peakVal = obj_new.peakVal
            obj_origin.sensor = obj_new.sensor
            obj_origin.state = "tracking"
            obj_origin.age += 1
            if method == 'none':
                obj_origin.x = obj_new.x
                obj_origin.y = obj_new.y
                obj_origin.dRange = obj_new.range - obj_origin.range
                obj_origin.range = obj_new.range
                obj_origin.dAzimuth = obj_new.azimuth - obj_origin.azimuth
                obj_origin.azimuth = obj_new.azimuth
            elif method == 'kalman':
                Z = np.array([[obj_new.azimuth], [obj_new.azimuth - obj_origin.azimuth], [obj_new.range]])
                proposalP = obj_origin.kalman.A.dot(obj_origin.kalman.P.dot(np.transpose(obj_origin.kalman.A))) + \
                            obj_origin.kalman.Q
                obj_origin.kalman.K = proposalP.dot(np.linalg.inv(proposalP + obj_origin.kalman.R))
                proposalX = obj_origin.kalman.A.dot(obj_origin.kalman.X)
                obj_origin.kalman.X = proposalX + obj_origin.kalman.K.dot(Z - proposalX)
                obj_origin.kalman.P = (np.eye(3) - obj_origin.kalman.K).dot(proposalP)
                if obj_origin.age > 2:
                    obj_origin.kalman.R = (obj_origin.age - 1) / (obj_origin.age - 2) * obj_origin.kalman.R + 1 / \
                                          obj_origin.age * (Z - obj_origin.kalman.X).dot(
                        np.transpose(Z - obj_origin.kalman.X))
                    obj_origin.kalman.Q = (obj_origin.age - 1) / (obj_origin.age - 2) * obj_origin.kalman.R + 1 / \
                                          obj_origin.age * (Z - obj_origin.kalman.X).dot(
                        np.transpose(Z - obj_origin.kalman.X))

                obj_origin.azimuth = obj_origin.kalman.X[0, 0]
                obj_origin.dAzimuth = obj_origin.kalman.X[1, 0]
                obj_origin.dRange = obj_origin.kalman.X[2, 0] - obj_origin.range
                obj_origin.range = obj_origin.kalman.X[2, 0]
                obj_origin.x = obj_origin.range * np.sin(np.radians(obj_origin.azimuth))
                obj_origin.y = obj_origin.range * np.cos(np.radians(obj_origin.azimuth))

        def trackObjects(self, method='none'):
            tempTrackingObject = self.trackingObjects.copy()
            if self.newList and not self.detectionFlag:
                Dtracking = np.empty([self.trackingObjects.__len__(), self.newList.__len__()])
                for j in range(self.trackingObjects.__len__()):
                    Dtracking[j,] = self.distFun(self.trackingObjects[j], self.newList)

                Dwaiting = np.empty([self.waitingList.__len__(), self.newList.__len__()])
                for j in range(self.waitingList.__len__()):
                    Dwaiting[j,] = self.distFun(self.waitingList[j], self.newList) * \
                                   self.constantObj.waitingListWeight

                Ddeleted = np.empty([self.topDeletedList.__len__(), self.newList.__len__()])
                for j in range(self.topDeletedList.__len__()):
                    Ddeleted[j,] = self.distFun(self.topDeletedList[j], self.newList) * \
                                   self.constantObj.topDeletedListWeight

                Dtotal = np.concatenate((Dtracking, Dwaiting), axis=0)
                Dtotal = np.concatenate((Dtotal, Ddeleted), axis=0)
                self.trackingObjects = []
                row_ind, col_ind = optimize.linear_sum_assignment(Dtotal)
                while len(row_ind):
                    matchIndex = [row_ind[0], col_ind[0]]
                    row_ind = np.delete(row_ind, 0)
                    col_ind = np.delete(col_ind, 0)
                    if Dtotal[matchIndex[0], matchIndex[1]] <= self.constantObj.maxDistance:
                        for i in range(len(row_ind)):
                            if row_ind[i] > matchIndex[0]:
                                row_ind[i] -= 1
                            if col_ind[i] > matchIndex[1]:
                                col_ind[i] -= 1
                        if matchIndex[0] < tempTrackingObject.__len__():
                            self.updateObject(tempTrackingObject[matchIndex[0]], self.newList[matchIndex[1]], method)
                            self.newList.pop(matchIndex[1])
                            self.trackingObjects.append(tempTrackingObject[matchIndex[0]])
                            tempTrackingObject.pop(matchIndex[0])
                        elif matchIndex[0] < tempTrackingObject.__len__() + self.waitingList.__len__():
                            tempId = matchIndex[0] - tempTrackingObject.__len__()
                            self.updateObject(self.waitingList[tempId], self.newList[matchIndex[1]], method)
                            self.trackingObjects.append(self.waitingList[tempId])
                            self.newList.pop(matchIndex[1])
                            self.waitingList.pop(tempId)
                            self.trackingObjects[-1].state = "new"
                        else:
                            tempId = matchIndex[0] - tempTrackingObject.__len__() - self.waitingList.__len__()
                            self.updateObject(self.topDeletedList[tempId], self.newList[matchIndex[1]], method)
                            self.trackingObjects.append(self.topDeletedList[tempId])
                            self.newList.pop(matchIndex[1])
                            self.topDeletedList.pop(tempId)
                            self.trackingObjects[-1].state = "new"
                        Dtotal = np.delete(Dtotal, matchIndex[0], axis=0)
                        Dtotal = np.delete(Dtotal, matchIndex[1], axis=1)
                self.constantObj.minpower = np.min([self.constantObj.powerUB, self.constantObj.minpower + 20])
            else:
                self.trackingObjects = []
                self.constantObj.minpower = self.constantObj.powerLB
            for obj in tempTrackingObject:
                obj.state = "deleted"
                self.topDeletedList.append(obj)

        def polling(self):
            if self.trackingObjects.__len__() > 2:
                dazimuthArray = np.array([obj.dAzimuth for obj in self.trackingObjects])
                mean = np.mean(dazimuthArray)
                std = np.std(dazimuthArray)
                probArray = stats.norm(mean, std).pdf(dazimuthArray)
                if self.trackingObjects[np.argmin(probArray)].state == "new":
                    self.trackingObjects.pop(np.argmin(probArray))

        def mKalman(self, external_source):
            def saturate(x, talorance=1):
                return np.max([np.min([x, talorance]), -talorance])
            if not self.detectionFlag:
                self.polling()
            self.normalizeScore()
            if self.trackingObjects:
                self.falseTrack = 0
                self.detectionFlag = False
                azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                    astype(np.float32)
                scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
                newEstimation = np.sum(azimuthArray * scoresArray).astype(np.float32)
                try:
                    tempOut = savgol_filter(np.diff(self.refAngleStack[-10:]), 9, 1)
                    tol = saturate(abs(tempOut[-1]) * 2.2, 1)
                except:
                    tol = 2
                self.kalmanObj.azimuthVelocity = saturate(newEstimation - self.kalmanObj.azimuth, tol)
                self.kalmanObj.azimuth += self.kalmanObj.azimuthVelocity
            else:
                self.falseTrack += 1
                if self.falseTrack > self.maxFalseTrack:
                    self.detectionFlag = True
                self.kalmanObj.azimuth += self.kalmanObj.azimuthVelocity

        def scoreUpdate(self):
            previousScore = np.sum([obj.score for obj in self.trackingObjects])
            for obj in self.trackingObjects:
                newRelativeAzimuth = obj.azimuth - self.kalmanObj.azimuth
                obj.score *= np.exp(-((newRelativeAzimuth - obj.relativeAzimuth) /
                                      self.constantObj.dAzimuthWeight) ** 2 -
                                    (obj.dRange / self.constantObj.dRangeWeight) ** 2)
                obj.relativeAzimuth = (newRelativeAzimuth + obj.age * obj.relativeAzimuth) / (obj.age + 1)
            self.kalmanObj.confidence = np.sum([obj.score for obj in self.trackingObjects]) / previousScore
            for obj in self.trackingObjects:
                obj.score = np.max([.2, obj.score])
            self.constellationMapBuild()

        def mergeList(self):
            self.waitingList = self.newList
            self.newList = []
            for obj in self.waitingList:
                obj.relativeAzimuth = obj.azimuth - self.kalmanObj.azimuth
            sorted(self.topDeletedList, key=attrgetter('age'), reverse=True)
            while self.topDeletedList.__len__() > 10:
                self.topDeletedList.pop(-1)
            for obj in self.topDeletedList:
                if obj.age > 0:
                    obj.age -= 1

        def normalizeScore(self):
            scoreSum = np.sum([obj.score for obj in self.trackingObjects])
            for obj in self.trackingObjects:
                obj.score = obj.score / scoreSum

        def constellationMapBuild(self):
            if self.constellation.__len__() < self.maxConstellationPoints:
                for obj in self.trackingObjects:
                    if not obj.id in [obj2.id for obj2 in self.constellation] and obj.age >= 5:
                        objtemp = copy.deepcopy(obj)
                        objtemp.azimuth = objtemp.relativeAzimuth
                        objtemp.dRange = 0
                        objtemp.dAzimuth = 0
                        self.constellation.append(objtemp)
                self.clustering(self.constellation, measure='score')
                if self.constellation.__len__() >= self.minConstellationPoints:
                    self.constellationMapIsReady = True

        def twoSetShiftAndDist(self, list1, list2, shift):
            from scipy.optimize import linear_sum_assignment
            list_temp = copy.deepcopy(list1)
            for obj in list_temp:
                obj.azimuth += shift
            if list_temp and list2:
                D = np.empty([list_temp.__len__(), list2.__len__()])
                for j in range(list_temp.__len__()):
                    D[j, ] = self.distFun(list_temp[j], list2, directivity=False)
                row_ind, col_ind = linear_sum_assignment(D)
                return D[row_ind, col_ind].sum()

        def detection(self):
            if self.newList and self.constellationMapIsReady and self.detectionFlag:
                res = 64
                angles = [-64, 64]
                dist   = [100, 100]
                dist[0] = self.twoSetShiftAndDist(self.constellation, self.newList, angles[0])
                dist[1] = self.twoSetShiftAndDist(self.constellation, self.newList, angles[1])
                while res >= .5:
                    if dist[0] >= dist[1]:
                        angles[0] = np.mean(angles)
                        dist[0]   = self.twoSetShiftAndDist(self.constellation, self.newList, angles[0])
                    elif dist[0] < dist[1]:
                        angles[1] = np.mean(angles)
                        dist[1]   = self.twoSetShiftAndDist(self.constellation, self.newList, angles[1])
                    res = res / 2
                if dist[1] <= dist[0]:
                    azimuth = angles[1]
                else:
                    azimuth = angles[0]
                self.trackingObjects = copy.deepcopy(self.constellation)
                for obj in self.trackingObjects:
                    obj.azimuth = obj.relativeAzimuth + azimuth

        def iterate(self, trigger='internal', external_source=np.nan):
            # if self.constellationMapIsReady:
            #     self.detectionFlag = True
            #     self.trackingObjects = []
            self.dataParser(external_source)
            self.clustering(self.newList)
            self.trackObjects(method='kalman')
            self.detection()
            self.mKalman(external_source)
            self.scoreUpdate()
            self.mergeList()
            self.normalizeScore()
            if trigger == 'internal':
                self.counter += 1
            elif trigger == 'external':
                pass
            else:
                sys.exit("trigger is not valid")

class imageProcessing:
        def __init__(self, FILE_NAME):
            self.FILE_NAME          = FILE_NAME
            self.MODEL_NAME         = 'inference_graph_FRCNN_V2'
            self.PATH_TO_CSV        = []
            self.PATH_TO_CKPT       = []
            self.PATH_TO_LABELS     = []
            self.PATH_TO_IMAGE      = []
            self.dataNum            = 0
            self.NUM_CLASSES        = 3
            self.objectDetectionObj = []
            self.objectTrackerObj   = []
            self.initialization()
            self.currentIndex       = 0
            self.IMAGE              = []
            self.boxes              = []
            self.driverML           = []
            self.passengerML        = []
            self.detctionFlag       = 0     # decimal of 00-> 11 from none to both - passenger driver
            self.updatePars         = True
            self.dimension          = self.dimensionCreat()
            self.hitchAngle         = []
            self.output             = False

        class objectDetectionModel:
            def __init__(self, PATH_TO_LABELS, NUM_CLASSES, PATH_TO_CKPT):
                label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
                categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                            max_num_classes=NUM_CLASSES,
                                                                            use_display_name=True)
                category_index = label_map_util.create_category_index(categories)
                detection_graph = tf.Graph()
                with detection_graph.as_default():
                    od_graph_def = tf.compat.v1.GraphDef()
                    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.compat.v1.Session(graph=detection_graph)
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                self.threshold = .8

        class objectTrackerModel:
            def __init__(self):
                self.TRACK_FLAG = False
                self.multitracker = cv2.MultiTracker_create()

        class dimensionCreat:
            def __init__(self):
                self.center         = []
                self.center.append(np.array([[0], [0.21443057]]))
                self.width          = []
                self.length         = []
                self.omega          = []
                self.passengerTran  = []
                self.driverTran     = []

        def initialization(self):
            self.CWD_PATH = os.getcwd()
            self.PATH_TO_CSV = os.path.join(self.CWD_PATH, self.FILE_NAME, self.FILE_NAME + '.csv')
            self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'label_map.pbtxt')
            self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
            self.objectDetectionObj = self.objectDetectionModel(self.PATH_TO_LABELS, self.NUM_CLASSES, self.PATH_TO_CKPT)
            self.objectTrackerObj = self.objectTrackerModel()
            dummy = pd.read_csv(self.PATH_TO_CSV)
            self.dataNum = len(dummy)

        def imageParser(self):
            index, time_stamp, angle, _, _, _, _, _, _, _, _, _, _ = pdf.readCSV(
                self.PATH_TO_CSV, self.currentIndex + 1)

            IMG_NAME = "img_%d.jpeg" % index
            PATH_TO_IMAGE = os.path.join(self.CWD_PATH, self.FILE_NAME, 'Figures/', IMG_NAME)
            self.IMAGE = cv2.imread(PATH_TO_IMAGE)
            self.IMAGE = imgFilter(self.IMAGE, u_lim=[0, 1], v_lim=[.7, 1])

        def detection(self):
            if not self.objectTrackerObj.TRACK_FLAG:
                image_expanded = np.expand_dims(self.IMAGE, axis=0)
                (boxes_ml, scores_ml, classes_ml, num) = self.objectDetectionObj.sess.run(
                    [self.objectDetectionObj.detection_boxes, self.objectDetectionObj.detection_scores,
                     self.objectDetectionObj.detection_classes, self.objectDetectionObj.num_detections],
                    feed_dict={self.objectDetectionObj.image_tensor: image_expanded})
                boxes_ml = boxes_ml.reshape(-1, 4)
                boxes_ml = boxes_ml[(scores_ml > self.objectDetectionObj.threshold).reshape(-1), ]
                self.boxes = pdf.tfcv_convertor(boxes_ml, self.IMAGE.shape[0:2], source='tf')
                for bbox in self.boxes:
                    self.objectTrackerObj.multitracker.add(cv2.TrackerMedianFlow_create(), self.IMAGE, bbox)
                self.boxes = pdf.tfcv_convertor(self.boxes, self.IMAGE.shape[0:2], source='cv')
                if self.boxes.shape[0] <= 1:
                    self.objectTrackerObj.TRACK_FLAG = False
                    self.objectTrackerObj.multitracker = cv2.MultiTracker_create()
                else:
                    self.objectTrackerObj.TRACK_FLAG = True

        def tracking(self):
            if self.objectTrackerObj.TRACK_FLAG:
                success, boxes_ml = self.objectTrackerObj.multitracker.update(self.IMAGE)
                self.boxes = pdf.tfcv_convertor(boxes_ml, self.IMAGE.shape[0:2], source='cv')
                if self.boxes.shape[0] <= 1:
                    self.objectTrackerObj.TRACK_FLAG = False
                    self.objectTrackerObj.multitracker = cv2.MultiTracker_create()
                else:
                    self.objectTrackerObj.TRACK_FLAG = True

        def hitchAngleEstimator(self):
            mid = pdf.undistortPoint(pdf.mid_detection(self.boxes), self.IMAGE.shape[:-1])
            mid[mid == 0] = np.nan
            mid = mid[~np.isnan(mid).any(axis=1)]
            cam_det = pdf.reversePinHole(mid, [0, 0, .78], -9, 0, self.IMAGE.shape[:-1])
            x_c = cam_det[0, ]
            y_c = cam_det[1, ]
            y_c = y_c[x_c.argsort()]
            x_c.sort()
            if len(x_c) == 2:
                self.detctionFlag = 3
                self.passengerML.append(np.array([[x_c[0]], [y_c[0]]]))
                self.driverML.append(np.array([[x_c[1]], [y_c[1]]]))
                if self.updatePars:
                    self.dimension.width.append(np.sqrt((x_c[0] - x_c[1])**2 + (y_c[0] - y_c[1])**2))
            elif len(x_c) == 1:
                if np.sum((np.array([[x_c[0]], [y_c[0]]]) - self.passengerML[-1]) ** 2) < \
                        np.sum((np.array([[x_c[0]], [y_c[0]]]) - self.driverML[-1]) ** 2):
                    self.detctionFlag = 2
                    self.passengerML.append(np.array([[x_c[0]], [y_c[0]]]))
                else:
                    self.detctionFlag = 1
                    self.driverML.append(np.array([[x_c[0]], [y_c[0]]]))
            else:
                self.detctionFlag = 0
            self.parEstimation()
            if list(np.binary_repr(self.detctionFlag, width=2))[0] == '1':
                temp_p = self.dimension.passengerTran.dot(self.passengerML[-1] - self.dimension.center[-1])
            else:
                temp_p = np.array([[np.nan], [np.nan]])
            if list(np.binary_repr(self.detctionFlag, width=2))[1] == '1':
                temp_d = self.dimension.driverTran.dot(self.driverML[-1] - self.dimension.center[-1])
            else:
                temp_d = np.array([[np.nan], [np.nan]])
            temp_mid = np.nanmean([temp_p, temp_d], axis=0)
            self.hitchAngle.append(np.degrees(np.arctan(temp_mid[0] / temp_mid[1])))

        def parEstimation(self):
            if self.updatePars:
                if self.detctionFlag > 0:
                    x = np.hstack(([i[0, 0] for i in self.passengerML], [i[0, 0] for i in self.driverML]))
                    y = np.hstack(([i[1, 0] for i in self.passengerML], [i[1, 0] for i in self.driverML]))
                    x = x[~np.isnan(x)]
                    y = y[~np.isnan(y)]
                    self.dimension.length.append(np.mean(np.sqrt((x - self.dimension.center[-1][0, 0])**2 +
                                                                 (y - self.dimension.center[-1][1, 0])**2)))
                    if np.max([i[0, 0] for i in self.passengerML]) - np.min([i[0, 0] for i in self.passengerML]) >= 1:
                        center, length, residual = centerCalc(x, y)
                        self.dimension.center.append(center)
                        self.dimension.length.append(length)
                        try:
                            if np.sqrt(np.sum(self.dimension.center[-1] - self.dimension.center[-2])**2) < .01:
                                self.updatePars = False
                        except:
                            pass
                    self.dimension.omega.append(np.arcsin(np.mean(self.dimension.width) / self.dimension.length[-1] / 2))
                    c, s = np.cos(self.dimension.omega[-1]), np.sin(self.dimension.omega[-1])
                    self.dimension.passengerTran =  np.array([[c, s], [-s, c]]).reshape(2, 2)
                    self.dimension.driverTran    =  np.array([[c, -s], [s, c]]).reshape(2, 2)

        def iterate(self, trigger='internal'):
            self.imageParser()
            self.detection()
            self.tracking()
            self.hitchAngleEstimator()
            if trigger == 'internal':
                self.currentIndex += 1
            elif trigger == 'external':
                pass
            else:
                sys.exit("trigger is not valid")

class fusion:
    def __init__(self, FILENAME, radar='enable', camera='enable'):
        if radar == 'enable':
            self.radarObj   = pointCloudProcessing(FILENAME, 0)
        else:
            self.radarObj   = []
        if camera == 'enable':
            self.cameraObj  = imageProcessing(FILENAME)
        else:
            self.cameraObj  = []
        self.hitchAngle     = []
        self.PATH_TO_RESULT = []
        self.FILE_NAME      = FILENAME
        self.RESULTS        = []
        self.counter        = 0
        self.dataNum        = 0
        self.initialization()
        if not self.cameraObj and not self.radarObj:
            sys.exit('no sensor selected')

    def initialization(self):
        CWD_PATH = os.getcwd()
        self.PATH_TO_CSV = os.path.join(CWD_PATH, self.FILE_NAME, self.FILE_NAME + '.csv')
        self.PATH_TO_RESULTS = os.path.join(CWD_PATH, self.FILE_NAME, 'RESULT-' + self.FILE_NAME + '.csv')
        dummy = pd.read_csv(self.PATH_TO_CSV)
        self.dataNum = len(dummy)
        self.RESULTS = pd.DataFrame(np.zeros([self.dataNum, 1]), columns=["Vernier"])

    def estimator(self):
        if self.radarObj and self.cameraObj:
            self.cameraObj.iterate(trigger='external')
            self.radarObj.iterate(trigger='external', external_source=self.cameraObj.hitchAngle[-1])
            # if self.radarObj.detector.state == 'collection':
            #     self.radarObj.detector.truth.append(self.cameraObj.hitchAngle[-1])
            self.hitchAngle.append(np.mean([self.radarObj.kalmanObj.azimuth, self.cameraObj.hitchAngle[-1]]))
        elif self.cameraObj:
            self.cameraObj.iterate(trigger='external')
        elif self.radarObj:
            self.radarObj.iterate(trigger='external')

    def iterate(self):
        self.estimator()
        if self.radarObj and self.cameraObj:
            self.RESULTS.loc[self.counter, "Radar"] = np.array(self.radarObj.kalmanObj.azimuth)
            self.RESULTS.loc[self.counter, "Camera"] = np.array(self.cameraObj.hitchAngle[-1])
            self.RESULTS.loc[self.counter, "Fusion"] = np.array(self.hitchAngle[-1])
            self.RESULTS.loc[self.counter, "Vernier"] = np.array(self.radarObj.refHitchAngle)
            if tadObj.cameraObj.objectTrackerObj.TRACK_FLAG:
                self.RESULTS.loc[self.counter, "Cam_Method"] = 'Tracking'
            else:
                self.RESULTS.loc[self.counter, "Cam_Method"] = 'Detection'
            if tadObj.cameraObj.updatePars:
                self.RESULTS.loc[self.counter, "Cam_Calib"] = 'on'
            else:
                self.RESULTS.loc[self.counter, "Cam_Calib"] = 'off'
            if tadObj.radarObj.trackingObjects:
                self.RESULTS.loc[self.counter, "Radar_Track_State"] = 'True'
            else:
                self.RESULTS.loc[self.counter, "Radar_Track_State"] = 'False'
            self.radarObj.counter += 1
            self.cameraObj.currentIndex += 1
        elif self.radarObj:
            self.RESULTS.loc[self.counter, "Radar"] = np.array(self.radarObj.kalmanObj.azimuth)
            self.RESULTS.loc[self.counter, "Fusion"] = np.nan
            self.RESULTS.loc[self.counter, "Camera"] = np.nan
            if tadObj.radarObj.trackingObjects:
                self.RESULTS.loc[self.counter, "Radar_Track_State"] = 'True'
            else:
                self.RESULTS.loc[self.counter, "Radar_Track_State"] = 'False'
            self.radarObj.counter += 1
        elif self.cameraObj:
            self.RESULTS.loc[self.counter, "Camera"] = np.array(self.cameraObj.hitchAngle[-1])
            self.RESULTS.loc[self.counter, "Fusion"] = np.nan
            self.RESULTS.loc[self.counter, "Radar"] = np.nan
            if tadObj.cameraObj.objectTrackerObj.TRACK_FLAG:
                self.RESULTS.loc[self.counter, "Cam_Method"] = 'Tracking'
            else:
                self.RESULTS.loc[self.counter, "Cam_Method"] = 'Detection'
            if tadObj.cameraObj.updatePars:
                self.RESULTS.loc[self.counter, "Cam_Calib"] = 'on'
            else:
                self.RESULTS.loc[self.counter, "Cam_Calib"] = 'off'
            self.cameraObj.currentIndex += 1
        self.counter += 1



if __name__ == '__main__':
    app = QtGui.QApplication([])
    pg.setConfigOption('background', 'w')
    win = pg.GraphicsWindow(title="2D scatter plot")
    p = win.addPlot()
    p.setXRange(-4, 4)
    p.setYRange(0, 4)
    p.setLabel('left', text='Y position (m)')
    p.setLabel('bottom', text='X position (m)')
    p.addLegend(size=None)
    radar = p.plot([], [], pen=None, symbol='o', color='blue', name='radar')
    camera = p.plot([], [], pen=None, symbol='x', color='red', name='camera')



    # FILE_NAME = '10-Aug-2020-16-29'
    # FILE_NAME = '28-Sep-2020-16-11'
    FILE_NAME = '28-Sep-2020-14-18'
    # FILE_NAME = '27-Oct-2020-11-23'
    # FILE_NAME = '09-Nov-2020-13-36'
    # FILE_NAME = '09-Nov-2020-14-24'

    tadObj = fusion(FILE_NAME)
    for INDEX in range(tadObj.dataNum):

        if INDEX == 287:
            print()
        tadObj.iterate()
        print(tadObj.counter, tadObj.cameraObj.hitchAngle[-1], tadObj.radarObj.kalmanObj.azimuth)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("Tracking\n", pd.DataFrame([tadObj.radarObj.toDict(obj) for obj in tadObj.radarObj.trackingObjects]))
            print('Detection', tadObj.radarObj.detectionFlag)
            print("Constellation\n", pd.DataFrame([tadObj.radarObj.toDict(obj) for obj in tadObj.radarObj.constellation]))


        xr = []
        yr = []
        xc = []
        yc = []

        for obj in tadObj.radarObj.trackingObjects:
            xr.append(obj.x)
            yr.append(obj.y)
        xc = [tadObj.cameraObj.driverML[-1][0, 0], tadObj.cameraObj.passengerML[-1][0, 0]]
        yc = [tadObj.cameraObj.driverML[-1][1, 0], tadObj.cameraObj.passengerML[-1][1, 0]]

        radar.setData(xr, yr)
        camera.setData(xc, yc)


        QtGui.QApplication.processEvents()

        time.sleep(0)
    while True:
        try:
            tadObj.RESULTS.to_csv(tadObj.PATH_TO_RESULTS, index=True)
            tadObj.RESULTS.to_csv("c:/TAD-Mojtaba/MATLAB/" + tadObj.FILE_NAME + '.csv', index=True)
            break
        except:
            print('Close that damn file idiot!!!')
            time.sleep(1)







        








