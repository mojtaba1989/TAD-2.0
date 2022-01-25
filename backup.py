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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")


def saturate(x, talorance=1):
    return np.max([np.min([x, talorance]), -talorance])


class pointCloudProcessing:
    def __init__(self, FILE_NAME, hitchAngle):
        self.trackingObjects = []
        self.newList = []
        self.waitingList = []
        self.topDeletedList = []
        self.center = [-0.03991714, 0.35]
        self.constantObj = self.constant()
        self.savgolObj = self.savgolMakeObj(hitchAngle)
        self.mpcObj = self.mpcMakeObj(hitchAngle)
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
            # filtering constants
            self.minRange = 1
            self.maxRange = 3
            self.azimuthWindow = 35
            # tracking constants
            self.rangeWeight = 100
            self.azimuthWeight = 1
            self.peakValWeight = 0
            self.waitingListWeight = 1
            self.topDeletedListWeight = 1
            self.directionmatchprize = .6
            self.maxDistance = np.sqrt(5 ** 2 + 8 ** 2)
            # scoring constants
            self.dRangeWeight = 4
            self.dAzimuthWeight = 1
            self.initialScore = 0.01

    class savgolMakeObj:
        def __init__(self, hitchAngle):
            self.azimuth = hitchAngle
            self.azimuthHist = []
            self.savgolWindow = 5
            self.savgolOrder = 3
            self.azimuthVelocity = 0
            self.azimuthVelocityHist = []
            self.velocityWindow = 5
            self.velocityOrder = 1
            self.confidence = 1

    class mpcMakeObj:
        def __init__(self, hitchAngle):
            self.azimuth = hitchAngle
            self.azimuthVelocity = 0
            self.dT = 1
            self.order = 5
            self.U = np.zeros(self.order)
            self.controller = 0
            self.maxAcceleration = .1
            self.bound = ((-self.maxAcceleration, self.maxAcceleration),)
            for i in range(1, self.order):
                self.bound += ((-self.maxAcceleration, self.maxAcceleration),)
            self.state = np.array([[self.azimuth], [self.azimuthVelocity]])
            self.A = np.array([[1, self.dT], [0, 1]])
            self.B = np.array([[.5 * self.dT ** 2], [self.dT]])
            self.C = np.array([[1, 0]])
            self.D = np.zeros([self.order, self.order])
            self.F = np.empty([self.order, 1])
            self.G = np.ones([self.order, 2])
            self.H = np.zeros([self.order, self.order])
            self.r = 1
            self.confidence = 1

            ## build matrices
            def h(n, dt):
                return (n - 1.5) * dt ** 2

            self.D[0:2, 0:2] = np.array([[1, -1], [0, 1]])
            for i in range(self.order):
                self.F[i, 0] = h((i + 2), self.dT)
            for i in range(self.order):
                self.G[i, 1] = (i + 1) * self.dT
            for r in range(1, self.order):
                for c in range(r):
                    self.H[r, c] = h(r - c + 1, self.dT)

    class kalmanMakeObj:
        def __init__(self, hitchAngle):
            self.azimuth = hitchAngle
            self.azimuthVelocity = 0
            self.confidence = 1
            self.X = np.array([[hitchAngle], [0]])
            self.A = np.array([[1, 1], [0, 1]])
            self.P = np.array([[1, 0], [0, 1]])
            self.K = np.eye(2)
            self.Q = np.array([[1, 0], [0, 4]])
            self.R = np.array([[1, 0], [0, 1]])

    def toDict(self, obj):
        return {
            "id": obj.id,
            "age": obj.age,
            # "x": obj.x,
            # "y": obj.y,
            # "peakVal": obj.peakVal,
            "range": obj.range,
            "azimuth": obj.azimuth,
            "score": obj.score,
            # "sensor": obj.sensor,
            # "state": obj.state,
            "relativeAzimuth": obj.relativeAzimuth,
            "dAzimuth": obj.dAzimuth
            # "dRange": obj.dRange
        }

    def setDirectory(self):
        CWD_PATH = os.getcwd()
        self.PATH_TO_CSV = os.path.join(CWD_PATH, self.FILE_NAME, self.FILE_NAME + '.csv')
        self.PATH_TO_RESULTS = os.path.join(CWD_PATH, self.FILE_NAME, 'RADAR-lined-' + self.FILE_NAME + '.csv')
        self.PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-lined-' + FILE_NAME + '.csv')
        dummy = pd.read_csv(self.PATH_TO_CSV)
        self.dataNum = len(dummy)
        self.result = pd.DataFrame(np.zeros([self.dataNum, 1]), columns=["Vernier"])

    def dataParser(self, methodObj):
        index, self.timeStamp, self.refHitchAngle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d = pdf.readCSV(
            self.PATH_TO_CSV, self.counter + 1)
        p_p = pdf.tm_f(p_p, .16, .84, .05, 15, 7, 'p')
        p_d = pdf.tm_f(p_d, .16, .84, .05, 20, 7, 'd')
        self.newList = []
        self.refAngleStack.append(self.refHitchAngle)
        for i in range(len(x_p)):
            self.newList.append(self.makeObj(p_p[0, i], p_p[1, i], peakVal_p[i],
                                             self.center, self.constantObj.initialScore, methodObj.azimuth, "passenger",
                                             self.currentID))
            self.currentID += 1
            if not np.abs(self.newList[-1].azimuth - methodObj.azimuth) <= self.constantObj.azimuthWindow or \
                    not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange) or not \
            self.newList[-1].peakVal > 100:
                self.newList.pop()
                self.currentID -= 1
        for i in range(len(x_d)):
            self.newList.append(self.makeObj(p_d[0, i], p_d[1, i], peakVal_d[i],
                                             self.center, self.constantObj.initialScore, methodObj.azimuth, "driver",
                                             self.currentID))
            self.currentID += 1
            if not np.abs(self.newList[-1].azimuth - methodObj.azimuth) <= self.constantObj.azimuthWindow or \
                    not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange):
                self.newList.pop()
                self.currentID -= 1

    def distFun(self, obj, list, methodObj, directivity=True):
        D = np.empty([3, list.__len__()])
        c = (self.constantObj.directionmatchprize + 1)/2
        s = (-self.constantObj.directionmatchprize + 1)/2
        for i in range(self.newList.__len__()):
            D[0, i] = (obj.range - list[i].range) * self.constantObj.rangeWeight
            D[1, i] = (obj.relativeAzimuth + methodObj.azimuth - list[i].azimuth) * self.constantObj.azimuthWeight
            D[2, i] = (obj.peakVal - list[i].peakVal) * self.constantObj.peakValWeight
        if methodObj.azimuthVelocity >= 0:
            Dangle = c - s * np.cos(np.arctan2(D[0, ], D[1, ]))
        else:
            Dangle = c - s * np.cos(np.pi - np.arctan2(D[0, ], D[1, ]))
        if directivity:
            Dsum   = np.sqrt(np.sum(np.abs(D) ** 2, axis=0)) * Dangle
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

    def clustering(self, methodObj):
        if self.newList:
            D = np.empty([self.newList.__len__(), self.newList.__len__()])
            for j in range(self.newList.__len__()):
                D[j, ] = self.distFun(self.newList[j], self.newList, methodObj, directivity=False)
            for j in range(self.newList.__len__()):
                D[j, j] = np.max(D)
            while np.min(D) == 0:
                matchIndex = np.unravel_index(D.argmin(), D.shape)
                self.newList.pop(matchIndex[1])
                D = np.delete(D, matchIndex[1], axis=1)

    def trackObjects(self, methodObj, method='none'):
        tempTrackingObject = self.trackingObjects
        if self.newList:
            Dtracking = np.empty([self.trackingObjects.__len__(), self.newList.__len__()])
            for j in range(self.trackingObjects.__len__()):
                Dtracking[j,] = self.distFun(self.trackingObjects[j], self.newList, methodObj)

            Dwaiting = np.empty([self.waitingList.__len__(), self.newList.__len__()])
            for j in range(self.waitingList.__len__()):
                Dwaiting[j,] = self.distFun(self.waitingList[j], self.newList, methodObj) * \
                               self.constantObj.waitingListWeight

            Ddeleted = np.empty([self.topDeletedList.__len__(), self.newList.__len__()])
            for j in range(self.topDeletedList.__len__()):
                Ddeleted[j,] = self.distFun(self.topDeletedList[j], self.newList, methodObj) * \
                               self.constantObj.topDeletedListWeight

            Dtotal = np.concatenate((Dtracking, Dwaiting), axis=0)
            Dtotal = np.concatenate((Dtotal, Ddeleted), axis=0)
            self.trackingObjects = []
            while not Dtotal.size == 0 and np.min(Dtotal) <= self.constantObj.maxDistance:
                matchIndex = np.unravel_index(Dtotal.argmin(), Dtotal.shape)
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
                else:
                    tempId = matchIndex[0] - tempTrackingObject.__len__() - self.waitingList.__len__()
                    self.updateObject(self.topDeletedList[tempId], self.newList[matchIndex[1]], method)
                    self.trackingObjects.append(self.topDeletedList[tempId])
                    self.newList.pop(matchIndex[1])
                    self.topDeletedList.pop(tempId)

                Dtotal = np.delete(Dtotal, matchIndex[0], axis=0)
                Dtotal = np.delete(Dtotal, matchIndex[1], axis=1)

        for obj in tempTrackingObject:
            obj.state = "deleted"
            self.topDeletedList.append(obj)

    def SAVGOL(self):
        self.normalizeScore()
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                astype(np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newHitchAngle = np.sum(azimuthArray * scoresArray).astype(np.float32)
        else:
            newHitchAngle = self.savgolObj.azimuth + self.savgolObj.azimuthVelocity

        self.savgolObj.azimuthHist.append(newHitchAngle)
        while self.savgolObj.azimuthHist.__len__() > self.savgolObj.savgolWindow:
            self.savgolObj.azimuthHist.pop(0)
        try:
            tempOut = savgol_filter(self.savgolObj.azimuthHist, self.savgolObj.savgolWindow,
                                    self.savgolObj.savgolOrder)
            self.savgolObj.azimuthHist = list(tempOut)
            estimatedHitchAngle = tempOut[-1]
            estimatedVelocity = estimatedHitchAngle - self.savgolObj.azimuth
            self.savgolObj.azimuthVelocityHist.append(estimatedVelocity)
            while self.savgolObj.azimuthVelocityHist.__len__() > self.savgolObj.velocityWindow:
                self.savgolObj.azimuthVelocityHist.pop(0)
            tempVelOut = savgol_filter(self.savgolObj.azimuthVelocityHist, self.savgolObj.velocityWindow,
                                       self.savgolObj.velocityOrder)
            self.savgolObj.azimuthVelocityHist = list(tempVelOut)
            finalVel = tempVelOut[-1]
            self.savgolObj.azimuthVelocity = finalVel
            self.savgolObj.azimuth = self.savgolObj.azimuth + finalVel
        except:
            self.savgolObj.azimuthVelocity = newHitchAngle - self.savgolObj.azimuth
            self.savgolObj.azimuth = newHitchAngle

    def mKalman(self):
        self.normalizeScore()
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                astype(np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newEstimation = np.sum(azimuthArray * scoresArray).astype(np.float32)
            try:
                tempOut = savgol_filter(np.diff(self.refAngleStack[-10:]), 9, 1)
                tol = saturate(abs(tempOut[-1]) * 2.2, 1)
            except:
                tol = 1
            self.kalmanObj.azimuthVelocity = saturate(newEstimation - self.kalmanObj.azimuth, tol)
        self.kalmanObj.azimuth += self.kalmanObj.azimuthVelocity

    def Kalman(self):
        self.normalizeScore()
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                astype(np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newEstimation = np.sum(azimuthArray * scoresArray).astype(np.float32)
            # if self.trackingObjects.__len__() >= 2:
            #     self.kalmanObj.R   = np.array([[np.var([azimuthArray]), 0],
            #                                    [0, np.var([obj.dAzimuth for obj in self.trackingObjects])]]) +\
            #                          np.random.random([2, 2])/100
            try:
                tempOut = savgol_filter(np.diff(self.refAngleStack[-10:]), 9, 1)
                tol = saturate(abs(tempOut[-1]) * 2.2, 1)
                # tol     = saturate(abs(tempOut)*3, 1)
            except:
                tol = 1
            Z = np.array([[newEstimation], [newEstimation - self.kalmanObj.azimuth]])
            proposalP = self.kalmanObj.A.dot(self.kalmanObj.P.dot(np.transpose(self.kalmanObj.A))) + \
                        self.kalmanObj.Q
            self.kalmanObj.K = proposalP.dot(np.linalg.inv(proposalP + self.kalmanObj.R))
            proposalX = self.kalmanObj.A.dot(self.kalmanObj.X)
            proposalX[1, 0] = saturate(proposalX[1, 0], tol)
            self.kalmanObj.X = proposalX + self.kalmanObj.K.dot(Z - proposalX)
            self.kalmanObj.P = (np.eye(2) - self.kalmanObj.K).dot(proposalP)
            # if self.counter > 2:
            #     self.kalmanObj.Q = (self.counter - 1) / (self.counter - 2) * self.kalmanObj.R + 1 / \
            #                           self.counter * (Z - self.kalmanObj.X).dot(
            #         np.transpose(Z - self.kalmanObj.X))

        self.kalmanObj.azimuth = self.kalmanObj.X[0, 0]
        self.kalmanObj.azimuthVelocity = self.kalmanObj.X[1, 0]

    def adaptiveSAVGOL(self):
        self.normalizeScore()
        try:
            if abs(self.refAngleStack[-1] - self.refAngleStack[-2]) > 0:
                self.savgolObj.savgolWindow = np.min([7, self.savgolObj.savgolWindow + 2])
                self.savgolObj.azimuthHist.insert(self.savgolObj.azimuthHist[0], 0)
                self.savgolObj.azimuthHist.insert(self.savgolObj.azimuthHist[0], 0)


            else:
                self.savgolObj.savgolWindow = 3
        except:
            pass
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                astype(np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newHitchAngle = np.sum(azimuthArray * scoresArray).astype(np.float32)
        else:
            newHitchAngle = self.savgolObj.azimuth + self.savgolObj.azimuthVelocity

        self.savgolObj.azimuthHist.append(newHitchAngle)
        while self.savgolObj.azimuthHist.__len__() > self.savgolObj.savgolWindow:
            self.savgolObj.azimuthHist.pop(0)
        try:
            tempOut = savgol_filter(self.savgolObj.azimuthHist, self.savgolObj.savgolWindow, self.savgolObj.savgolOrder)
            self.savgolObj.azimuthHist = list(tempOut)
            estimatedHitchAngle = tempOut[-1]
            estimatedVelocity = estimatedHitchAngle - self.savgolObj.azimuth
        except:
            estimatedVelocity = self.savgolObj.azimuthVelocity

        self.savgolObj.azimuthVelocityHist.append(estimatedVelocity)
        while self.savgolObj.azimuthVelocityHist.__len__() > self.savgolObj.velocityWindow:
            self.savgolObj.azimuthVelocityHist.pop(0)
        try:
            tempVelOut = savgol_filter(self.savgolObj.azimuthVelocityHist, self.savgolObj.velocityWindow,
                                       self.savgolObj.velocityOrder)
            self.savgolObj.azimuthVelocityHist = list(tempVelOut)
            finalVel = tempVelOut[-1]
            if finalVel > 0:
                finalVel = np.min([1.5, finalVel])
            else:
                finalVel = np.max([-1.5, finalVel])
        except:
            finalVel = estimatedVelocity
        self.savgolObj.azimuthVelocity = finalVel
        self.savgolObj.azimuth = self.savgolObj.azimuth + finalVel
        # self.savgolObj.azimuthHist[0] = estimatedHitchAngle

    def MPC(self):
        self.normalizeScore()
        self.mpcObj.state = self.mpcObj.A.dot(self.mpcObj.state) + self.mpcObj.B * self.mpcObj.controller
        tempState = self.mpcObj.state
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]). \
                astype(np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newHitchAngle = np.sum(azimuthArray * scoresArray).astype(np.float32)
            tempState[0, 0] = newHitchAngle

        def jFun(U):
            Utemp = U.reshape([-1, 1])
            Y = self.mpcObj.G.dot(tempState) + self.mpcObj.H.dot(Utemp) + self.mpcObj.F * self.mpcObj.controller
            out = np.transpose(Y).dot(Y) + self.mpcObj.r * np.transpose(Utemp).dot(
                np.transpose(self.mpcObj.D).dot(self.mpcObj.D.dot(Utemp)))
            return out.reshape(-1)

        res = optimize.minimize(jFun, self.mpcObj.U, bounds=self.mpcObj.bound)
        self.mpcObj.U = res.x
        self.mpcObj.controller = self.mpcObj.U[0]
        self.mpcObj.azimuth = self.mpcObj.state[0, 0]

    def scoreUpdate(self, methodObj):
        previousScore = np.sum([obj.score for obj in self.trackingObjects])
        for obj in self.trackingObjects:
            newRelativeAzimuth = obj.azimuth - methodObj.azimuth
            obj.score *= np.exp(-((newRelativeAzimuth - obj.relativeAzimuth) /
                                  self.constantObj.dAzimuthWeight) ** 2 -
                                (obj.dRange / self.constantObj.dRangeWeight) ** 2)
            obj.relativeAzimuth = (newRelativeAzimuth + obj.age * obj.relativeAzimuth) / (obj.age + 1)
        methodObj.confidence = np.sum([obj.score for obj in self.trackingObjects]) / previousScore
        for obj in self.trackingObjects:
            obj.score = np.max([.2, obj.score])

    def mergeList(self, methodObj):
        self.waitingList = self.newList
        self.newList = []
        for obj in self.waitingList:
            obj.relativeAzimuth = obj.azimuth - methodObj.azimuth
        sorted(self.trackingObjects, key=attrgetter('score'), reverse=True)
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

    def iterate(self, methodObj):
        self.dataParser(methodObj)
        # self.clustering(methodObj)
        self.trackObjects(methodObj, method='kalman')
        if methodObj == self.savgolObj:
            self.SAVGOL()
        elif methodObj == self.mpcObj:
            self.MPC()
        elif methodObj == self.kalmanObj:
            self.mKalman()
        self.scoreUpdate(methodObj)
        self.mergeList(methodObj)
        self.normalizeScore()
        self.result.loc[self.counter, "Vernier"] = np.array(self.refHitchAngle)
        self.result.loc[self.counter, "Radar"] = methodObj.azimuth
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
    s_d_new = p.plot([], [], pen=None, symbol='o', color='blue', name='Driver-new')
    s_d_tracking = p.plot([], [], pen=None, symbol='s', color='red', name='Driver-tracking')
    s_p_new = p.plot([], [], pen=None, symbol='x', color='blue', name='Passenger-new')
    s_p_tracking = p.plot([], [], pen=None, symbol='t', color='red', name='Passenger-tracking')

    FILE_NAME = '10-Aug-2020-17-47'
    # FILE_NAME = '10-Aug-2020-16-29'
    # FILE_NAME = '28-Sep-2020-16-11'
    # FILE_NAME = '28-Sep-2020-14-18'
    radarObj = pointCloudProcessing(FILE_NAME, -.5)

    for INDEX in range(radarObj.dataNum):
        print(INDEX, radarObj.refHitchAngle, radarObj.mpcObj.azimuth, radarObj.mpcObj.controller)
        # print(radarObj.savgolObj.gain, radarObj.savgolObj.var)

        if INDEX == 49:
            print()
        # radarObj.iterate(radarObj.mpcObj)
        radarObj.iterate(radarObj.kalmanObj)

        #
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd.DataFrame([radarObj.toDict(obj) for obj in radarObj.trackingObjects]))
            # print(pd.DataFrame([radarObj.toDict(obj) for obj in radarObj.waitingList]))

        xpn = []
        ypn = []
        xdn = []
        ydn = []
        xpt = []
        ypt = []
        xdt = []
        ydt = []
        for obj in radarObj.trackingObjects:
            if obj.sensor == "driver":
                if obj.state == "new":
                    xdn.append(obj.x)
                    ydn.append(obj.y)
                else:
                    xdt.append(obj.x)
                    ydt.append(obj.y)
            else:
                if obj.state == "new":
                    xpn.append(obj.x)
                    ypn.append(obj.y)
                else:
                    xpt.append(obj.x)
                    ypt.append(obj.y)
        s_d_new.setData(xdn, ydn)
        s_p_new.setData(xpn, ypn)
        s_d_tracking.setData(xdt, ydt)
        s_p_tracking.setData(xpt, ypt)

        QtGui.QApplication.processEvents()

        time.sleep(0.0)

    while True:
        try:
            radarObj.result.to_csv(radarObj.PATH_TO_RESULTS, index=True)
            radarObj.result.to_csv("c:/TAD-Mojtaba/MATLAB/" + radarObj.FILE_NAME + '.csv', index=True)
            break
        except:
            print('Close that damn file idiot!!!')
            time.sleep(1)
    win.close()
    win2 = pg.GraphicsWindow(title="Result")
    p2 = win2.addPlot()
    radar = p2.plot(radarObj.result.Radar, pen='r', name='radar')
    Ver = p2.plot(radarObj.result.Vernier, pen='b', name='vernier')
    QtGui.QApplication.processEvents()
    time.sleep(1000)



