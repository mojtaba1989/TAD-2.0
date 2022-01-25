import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import python_data_fusion as pdf



class imageProcessing:
        def __init__(self, FILE_NAME, initial=0):
            self.FILE_NAME          = FILE_NAME
            self.PATH_TO_CSV        = {}
            self.PATH_TO_RESULTS    = {}
            self.PATH_TO_IMG_RESULTS= {}
            self.IMG_NAME           = {}
            self.PATH_TO_IMAGE      = []
            self.dataNum            = 0
            self.lineObj            = []
            self.currentIndex       = 0
            self.IMAGE              = 0
            self.hitchAngle         = initial
            self.refHitchAngle      = []
            self.result             = {}
            self.output             = True
            self.trackingFlag       = False
            self.center             = [[-0.007], [.232]]
            self.initialization()
            self.mask = np.zeros((960,1280), dtype="uint8")
            cv2.circle(self.mask, (640, 1160), 600, 255, -1)

        class lineObjCreate():
            def __init__(self, x1, y1, x2, y2, image, hitchangle, center):
                self.xd1 = x1
                self.xd2 = x2
                self.yd1 = y1
                self.yd2 = y2
                temp = pdf.normalize(np.array([[x1, y1], [x2, y2]]), image.shape[:-1])
                # temp = pdf.undistortPoint(temp, image.shape[:-1])
                cam_det = pdf.reversePinHole(temp, [0, 0, .78], -9, 0, image.shape[:-1])
                self.x1 = cam_det[0,0] - center[0]
                self.y1 = cam_det[1,0] - center[1]
                self.x2 = cam_det[0,1] - center[0]
                self.y2 = cam_det[1,1] - center[1]
                self.deleteFlag = np.any(temp == 0)
                try:
                    mainAngle = np.degrees(np.arctan((self.x1-self.x2)/(self.y1-self.y2)))
                    temp = [-90, 0, 90] + mainAngle
                    self.proposal = self.find_nearest(temp, hitchangle)
                except:
                    pass

            def find_nearest(self, array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]

        def initialization(self):
            self.CWD_PATH = os.getcwd()
            self.PATH_TO_CSV = os.path.join(self.CWD_PATH, self.FILE_NAME, self.FILE_NAME + '.csv')
            self.PATH_TO_RESULTS = os.path.join(self.CWD_PATH, self.FILE_NAME, self.FILE_NAME +'-Hough'+ '.csv')
            dummy = pd.read_csv(self.PATH_TO_CSV)
            self.dataNum = len(dummy)
            self.result = pd.DataFrame(np.zeros([self.dataNum, 3]), columns=["Vernier", "Cam", "time"])

        def imageParser(self):
            index, time_stamp, self.refHitchAngle, _, _, _, _, _, _, _, _, _, _ = pdf.readCSV(
                self.PATH_TO_CSV, self.currentIndex + 1)

            self.IMG_NAME = "img_%d.jpeg" % index
            PATH_TO_IMAGE = os.path.join(self.CWD_PATH, self.FILE_NAME, 'Figures/', self.IMG_NAME)
            self.PATH_TO_IMG_RESULTS = os.path.join(self.CWD_PATH, self.FILE_NAME, 'Figures/', '000'+self.IMG_NAME)
            self.IMAGE = cv2.imread(PATH_TO_IMAGE)

        def sobel(self, src):
            scale = 1
            delta = 0
            ddepth = cv2.CV_16S
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)

            return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        def detection(self):
            imageUndistored = pdf.undistort_org(self.IMAGE)
            gray = cv2.cvtColor(imageUndistored, cv2.COLOR_BGR2GRAY)
            # edges = self.sobel(imageUndistored)
            edges = cv2.Canny(gray, 60, 70, apertureSize=3)
            edges[0:600, :] = 0
            # edges = cv2.bitwise_and(edges, edges, mask=self.mask)
            minLineLength = 300
            maxLineGap = 5
            lines = cv2.HoughLinesP(edges, .1, np.pi / 180, 20, minLineLength, maxLineGap)
            self.lineObj = []
            try:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        self.lineObj.append(self.lineObjCreate(x1, y1, x2, y2, imageUndistored, self.hitchAngle, self.center))
                        if self.lineObj[-1].deleteFlag:
                            self.lineObj.pop()
            except:
                pass
            rgbEdge = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            if self.output:
                try:
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(rgbEdge, (x1, y1), (x2, y2), (255, 0, 0), 5)
                except:
                    pass
                images = np.hstack((imageUndistored, rgbEdge))
                cv2.imshow('Object detector', cv2.resize(images, (1200, 450)))
                if np.mod(self.currentIndex, 5)==0:
                    cv2.imwrite(self.PATH_TO_IMG_RESULTS, rgbEdge)
                cv2.waitKey(1)

        def hitchAngleEstimator(self):
            from scipy.stats import norm
            if self.lineObj:
                proposals = [obj.proposal for obj in self.lineObj]
                p_proposals = norm.pdf(proposals, self.hitchAngle, 2)
                p_proposals = p_proposals * (p_proposals > .01)
                try:
                    if np.sum(p_proposals) > 0:
                        p_proposals = p_proposals / np.sum(p_proposals)
                        self.hitchAngle = np.sum(proposals * p_proposals)
                except:
                    pass
                # self.hitchAngle.append(np.mean(proposals))

        def save(self):
            while True:
                try:
                    self.result.to_csv(self.PATH_TO_RESULTS, index=True)
                    self.result.to_csv('c:/TAD-Mojtaba/MATLAB/' + self.FILE_NAME + '.csv', index=True)
                    break
                except:
                    print('Close that damn file idiot!!!')
                    time.sleep(1)

        def iterate(self):
            tstart = time.time()
            self.imageParser()
            self.detection()
            self.hitchAngleEstimator()
            tend = time.time()
            self.result.loc[self.currentIndex, "Vernier"] = np.array(self.refHitchAngle)
            self.result.loc[self.currentIndex, "Cam"] = self.hitchAngle
            self.result.loc[self.currentIndex, "time"] = tend-tstart
            self.currentIndex += 1

        def exe(self):
            for _ in range(self.dataNum):
                self.iterate()
            self.save()






count = 0
if __name__ == '__main__':
    # FILE_NAME = '28-Sep-2020-15-59'
    # FILE_NAME = '02-Nov-2020-13-36'
    # FILE_NAME = '01-Oct-2019-14-04'
    FILE_NAME = '09-Nov-2020-13-36'

    camObj = imageProcessing(FILE_NAME, 0)
    camObj.exe()

