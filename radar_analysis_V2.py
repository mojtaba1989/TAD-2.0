import numpy as np
import os
import sys
import python_data_fusion as pdf
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pandas as pd
import time
from operator import attrgetter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")


class pointCloudProcessing:
    def __init__(self, FILE_NAME, hitchAngle):
        self.trackingObjects = []
        self.newList = []
        self.center = [-0.03991714, 0.27289268]
        self.constantObj = self.constant()
        self.kalmanObj = self.kalman(hitchAngle)
        self.counter = 0
        self.PATH_TO_CSV = {}
        self.FILE_NAME = FILE_NAME
        self.dataNum = 0
        self.result = {}
        self.PATH_TO_RESULTS = {}
        self.timeStamp = 0
        self.refHitchAngle = 0
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
            self.sensorMissmatchWeitght = 10
            self.waitingListWeight = 3
            self.topDeletedListWeight = 1.1
            # scoring constants
            self.dRangeWeight = 1
            self.dAzimuthWeight = 10
            self.initialScore = 0.001

    class kalman:
        def __init__(self, hitchAngle):
            self.azimuth = hitchAngle
            self.azimuthVelocity = 0
            self.var = 10
            self.gain = 1

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

    def dataParser(self):
        index, self.timeStamp, self.refHitchAngle, x_p, y_p, range_p, peakVal_p, x_d, y_d, range_d, peakVal_d, p_p, p_d = pdf.readCSV(
            self.PATH_TO_CSV, self.counter + 1)
        p_p = pdf.tm_f(p_p, .16, .84, .05, 15, 7, 'p')
        p_d = pdf.tm_f(p_d, .16, .84, .05, 20, 7, 'd')
        self.newList = []
        for i in range(len(x_p)):
            self.newList.append(self.makeObj(p_p[0, i], p_p[1, i], peakVal_p[i],
                                             self.center, self.constantObj.initialScore, self.kalmanObj.azimuth,
                                             "passenger", self.currentID))
            self.currentID += 1
            if not np.abs(self.newList[-1].azimuth - self.kalmanObj.azimuth) <= self.constantObj.azimuthWindow or \
                    not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange):
                self.newList.pop()
                self.currentID -= 1
        for i in range(len(x_d)):
            self.newList.append(self.makeObj(p_d[0, i], p_d[1, i], peakVal_d[i],
                                             self.center, self.constantObj.initialScore, self.kalmanObj.azimuth,
                                             "driver", self.currentID))
            self.currentID += 1
            if not np.abs(self.newList[-1].azimuth - self.kalmanObj.azimuth) <= self.constantObj.azimuthWindow or \
                    not (self.constantObj.minRange <= self.newList[-1].range <= self.constantObj.maxRange):
                self.newList.pop()
                self.currentID -= 1

    def trackObjects(self):
        tempTrackingObject = self.trackingObjects
        if self.newList:
            Dtracking = np.empty([self.trackingObjects.__len__(), self.newList.__len__()])
            for j in range(self.trackingObjects.__len__()):
                newObjScore = np.empty([3, self.newList.__len__()])
                for i in range(self.newList.__len__()):
                    newObjScore[0, i] = np.abs(
                        self.trackingObjects[j].range - self.newList[i].range) * self.constantObj.rangeWeight
                    newObjScore[1, i] = np.abs(self.trackingObjects[j].azimuth - self.newList[i].azimuth) \
                                        * self.constantObj.azimuthWeight
                    newObjScore[2, i] = np.abs(
                        self.trackingObjects[j].peakVal - self.newList[i].peakVal) * self.constantObj.peakValWeight
                Dtracking[j,] = np.sum(newObjScore, axis=0) / self.trackingObjects[j].score
                if not self.newList[i].sensor == self.trackingObjects[j].sensor:
                    Dtracking[j,] *= self.constantObj.sensorMissmatchWeitght

            Dtotal = Dtracking
            self.trackingObjects = []
            while not Dtotal.size == 0:
                matchIndex = np.unravel_index(Dtotal.argmin(), Dtotal.shape)
                tempTrackingObject[matchIndex[0]].x = self.newList[matchIndex[1]].x
                tempTrackingObject[matchIndex[0]].y = self.newList[matchIndex[1]].y
                tempTrackingObject[matchIndex[0]].dRange = self.newList[matchIndex[1]].range - tempTrackingObject[
                    matchIndex[0]].range
                tempTrackingObject[matchIndex[0]].range = self.newList[matchIndex[1]].range
                tempTrackingObject[matchIndex[0]].dAzimuth = self.newList[matchIndex[1]].azimuth - \
                                                             tempTrackingObject[matchIndex[0]].azimuth
                tempTrackingObject[matchIndex[0]].azimuth = self.newList[matchIndex[1]].azimuth
                tempTrackingObject[matchIndex[0]].peakVal = self.newList[matchIndex[1]].peakVal
                tempTrackingObject[matchIndex[0]].sensor = self.newList[matchIndex[1]].sensor
                tempTrackingObject[matchIndex[0]].state = "tracking"
                tempTrackingObject[matchIndex[0]].age += 1
                self.newList.pop(matchIndex[1])
                self.trackingObjects.append(tempTrackingObject[matchIndex[0]])
                tempTrackingObject.pop(matchIndex[0])

                Dtotal = np.delete(Dtotal, matchIndex[0], axis=0)
                Dtotal = np.delete(Dtotal, matchIndex[1], axis=1)

        while True:
            dAzimuthMat = [obj.dAzimuth for obj in self.trackingObjects]
            if np.var(dAzimuthMat) > self.kalmanObj.var:

        for obj in tempTrackingObject:
            obj.state = "virtual"
            self.topDeletedList.append(obj)

    def hitchAngleUpdate(self):
        self.normalizeScore()
        if self.trackingObjects:
            azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]).astype(
                np.float32)
            scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
            newHitchAngle = np.sum(azimuthArray * scoresArray).astype(np.float32)
            stdOfAzimuthArrzy = np.sqrt(np.sum(((scoresArray * (azimuthArray - newHitchAngle) ** 2))))
            deviation = (azimuthArray - newHitchAngle) / stdOfAzimuthArrzy
            kicklist = abs(deviation) > 3 * stdOfAzimuthArrzy + 1
            if np.any(kicklist):
                tempTrackingObjects = []
                for i in range(self.trackingObjects.__len__()):
                    if not kicklist[i]:
                        tempTrackingObjects.append(self.trackingObjects[i])
                self.trackingObjects = tempTrackingObjects
                self.normalizeScore()
                azimuthArray = np.array([obj.azimuth - obj.relativeAzimuth for obj in self.trackingObjects]).astype(
                    np.float32)
                scoresArray = np.array([obj.score for obj in self.trackingObjects]).astype(np.float32)
                newHitchAngle = np.sum(azimuthArray * scoresArray).astype(np.float32)
            self.kalmanObj.azimuthVelocity = newHitchAngle - self.kalmanObj.azimuth
            self.kalmanObj.azimuth = newHitchAngle

    def scoreUpdate(self):
        for obj in self.trackingObjects:
            newRelativeAzimuth = obj.azimuth - self.kalmanObj.azimuth
            obj.score = obj.score * np.exp(
                -((newRelativeAzimuth - obj.relativeAzimuth) / self.constantObj.dAzimuthWeight) ** 2 - (
                            obj.dRange / self.constantObj.dRangeWeight) ** 2)
            obj.relativeAzimuth = (newRelativeAzimuth + obj.age * obj.relativeAzimuth) / (obj.age + 1)

    def mergeList(self):
        self.waitingList = self.newList
        self.newList = []
        sorted(self.trackingObjects, key=attrgetter('score'), reverse=True)
        sorted(self.topDeletedList, key=attrgetter('age'), reverse=True)
        while self.topDeletedList.__len__() > 1:
            self.topDeletedList.pop(-1)
        for obj in self.topDeletedList:
            if obj.age > 0:
                obj.age -= 1

    def normalizeScore(self):
        scoreSum = np.sum([obj.score for obj in self.trackingObjects])
        for obj in self.trackingObjects:
            obj.score = obj.score / scoreSum

    def iterate(self):
        self.dataParser()
        self.trackObjects()
        self.hitchAngleUpdate()
        self.scoreUpdate()
        self.mergeList()
        self.normalizeScore()
        self.result.loc[self.counter, "Vernier"] = np.array(self.refHitchAngle)
        self.result.loc[self.counter, "Radar"] = self.kalmanObj.azimuth
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

    FILE_NAME = '10-Aug-2020-16-29'
    radarObj = pointCloudProcessing(FILE_NAME, 0)

    for INDEX in range(radarObj.dataNum):
        print(INDEX, radarObj.refHitchAngle)
        if INDEX == 449:
            print()
        radarObj.iterate()

        #
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd.DataFrame([radarObj.toDict(obj) for obj in radarObj.trackingObjects]))
        #     print(pd.DataFrame([radarObj.toDict(obj) for obj in radarObj.waitingList]))

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



