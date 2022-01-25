# Libraries
import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from  object_detection.utils import label_map_util
from scipy import optimize


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

def centerCalc(objects, initial_center=(0, 0)):
    from scipy import optimize
    def calc_R(xc, yc):
        for obj in objects:
            obj.R = np.sqrt((obj.y - yc) ** 2 + (obj.x - xc) ** 2)
            obj.R_error = obj.R - obj.R.mean()
            obj.df2b_dc = np.empty((2, len(obj.x)))
            obj.df2b_dc[0] = (xc - obj.x) / obj.R
            obj.df2b_dc[1] = (yc - obj.y) / obj.R
            obj.df2b_dc = obj.df2b_dc - obj.df2b_dc.mean(axis=1)[:, np.newaxis]

    def f(c):
        calc_R(*c)
        for obj in objects:
            try:
                error = np.concatenate((error, obj.R_error), axis=1)
            except:
                error = obj.R_error
        return error

    def Df_2b():
        for obj in objects:
            try:
                df2b_tot = np.concatenate((df2b_tot, obj.df2b_dc), axis=1)
            except:
                df2b_tot = obj.df2b_dc
        return df2b_tot

    center, ier = optimize.leastsq(f, initial_center, Dfun=Df_2b, col_deriv=True, maxfev=400)
    xc, yc = center
    return center.reshape(2, 1)

class hitchBallPosition:
    def __init__(self, FILE_NAME):
        self.FILE_NAME          = FILE_NAME
        self.MODEL_NAME         = 'inference_graph_FRCNN_V2'
        self.PATH_TO_CSV        = []
        self.PATH_TO_CKPT       = []
        self.PATH_TO_LABELS     = []
        self.PATH_TO_IMAGE      = []
        self.hitchBall          = (0, 0)
        self.dataNum            = 0
        self.objectDetectionObj = []
        self.objectTrackerObj   = []
        self.currentIndex       = 0
        self.IMAGE              = []

    class objectDetectionModel:
        def __init__(self, PATH_TO_LABELS, PATH_TO_CKPT):
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                        max_num_classes=3,
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
        index, _, _, _, _, _, _, _, _, _, _, _, _ = pdf.readCSV(
            self.PATH_TO_CSV, self.currentIndex + 1)

        IMG_NAME = "img_%d.jpeg" % index
        PATH_TO_IMAGE = os.path.join(self.CWD_PATH, self.FILE_NAME, 'Figures/', IMG_NAME)
        self.IMAGE = cv2.imread(PATH_TO_IMAGE)

    def detection(self):
        if not self.objectTrackerObj.TRACK_FLAG:
            image_expanded = np.expand_dims(self.IMAGE, axis=0)
            (boxes_ml, scores_ml, classes_ml, num) = self.objectDetectionObj.sess.run(
                [self.objectDetectionObj.detection_boxes, self.objectDetectionObj.detection_scores,
                 self.objectDetectionObj.detection_classes, self.objectDetectionObj.num_detections],
                feed_dict={self.objectDetectionObj.image_tensor: image_expanded})
            boxes_ml = boxes_ml.reshape(-1, 4)
            boxes_ml = boxes_ml[(scores_ml > self.objectDetectionObj.threshold).reshape(-1),]
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

    def parEstimation(self):
        if self.updatePars:
            if self.detctionFlag > 0:
                x = np.hstack(([i[0, 0] for i in self.passengerML], [i[0, 0] for i in self.driverML]))
                y = np.hstack(([i[1, 0] for i in self.passengerML], [i[1, 0] for i in self.driverML]))
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                self.dimension.length.append(np.mean(np.sqrt((x - self.dimension.center[-1][0, 0]) ** 2 +
                                                             (y - self.dimension.center[-1][1, 0]) ** 2)))
                if np.max([i[0, 0] for i in self.passengerML]) - np.min([i[0, 0] for i in self.passengerML]) >= 1:
                    center, length, residual = centerCalc(x, y)
                    self.dimension.center.append(center)
                    self.dimension.length.append(length)
                    try:
                        if np.sqrt(np.sum(self.dimension.center[-1] - self.dimension.center[-2]) ** 2) < .01:
                            self.updatePars = False
                    except:
                        pass
                self.dimension.omega.append(np.arcsin(np.mean(self.dimension.width) / self.dimension.length[-1] / 2))
                c, s = np.cos(self.dimension.omega[-1]), np.sin(self.dimension.omega[-1])
                self.dimension.passengerTran = np.array([[c, s], [-s, c]]).reshape(2, 2)
                self.dimension.driverTran = np.array([[c, -s], [s, c]]).reshape(2, 2)

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


