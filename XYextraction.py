import numpy as np
import os
import sys
import glob
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import python_data_fusion as pdf
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")

if __name__ == '__main__':
    for idx, arg in enumerate(sys.argv):
        if arg in ['--file', '-f']:
            FILE_NAME = str(sys.argv[idx+1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            is_error = True

    for idx, arg in enumerate(sys.argv):
        if arg in ['--graph', '-g']:
            OUTPUT = 'graph'
            del sys.argv[idx]
        elif arg in ['--image', '-i']:
            OUTPUT = 'image'
            del sys.argv[idx]
        else:
            is_error = True

    CWD_PATH = os.getcwd()
    PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME+'.csv')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    MODEL_NAME = 'inference_graph_FRCNN'
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # making output csv
    num = len(glob.glob(os.path.join(CWD_PATH, FILE_NAME, 'Figures/*.jpeg')))
    PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, 'XY.csv')
    Results = pd.DataFrame(np.zeros([num, 2]), columns=["X", "Y"])

    for idx, arg in enumerate(sys.argv):
        if arg in ['--threshold', '-t']:
            threshold = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]
        else:
            threshold = False

    if len(sys.argv) != 1:
        is_error = True
    else:
        for arg in sys.argv:
            if arg.startswith('-'):
                is_error = True

    NUM_CLASSES = 3
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Lamp detection
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Lamp tracking
    TRACK_FLAG = False
    multitracker = cv2.MultiTracker_create()
    center = np.array([[0], [0]])
    p_c_old = np.nan
    d_c_old = np.nan
    # Main Loop
    for INDEX in range(num):
        print(INDEX)
        tstart = time.time()
        index, time_stamp, angle = pdf.readCSV(
            PATH_TO_CSV, INDEX)

        IMG_NAME = "img_%d.jpeg" % index
        PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
        image = cv2.imread(PATH_TO_IMAGE)

        if not TRACK_FLAG:
            image_expanded = np.expand_dims(image, axis=0)
            (boxes_ml, scores_ml, classes_ml, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            boxes_ml = boxes_ml.reshape(-1, 4)
            if threshold is not False:
                boxes_ml   = boxes_ml[(scores_ml > threshold).reshape(-1), ]
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='tf')
            for bbox in boxes_ml:
                multitracker.add(cv2.TrackerMedianFlow_create(), image, bbox)
            TRACK_FLAG = True
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')

        else:
            success, boxes_ml = multitracker.update(image)
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')

        mid     = pdf.undistortPoint(pdf.mid_detection(boxes_ml), image.shape[:-1])
        mid[mid == 0] = np.nan
        mid = mid[~np.isnan(mid).any(axis=1)]

        cam_det = pdf.reversePinHole(mid, [0, 0, .9], -8, 0, image.shape[:-1])
        # Image processing:
        x_c = cam_det[0, ]
        y_c = cam_det[1, ]
        y_c = y_c[x_c.argsort()]
        x_c.sort()

        if len(x_c) == 2:
            missed_flag = False
            p_c = np.array([[x_c[0]], [y_c[0]]])
            d_c = np.array([[x_c[1]], [y_c[1]]])
            p_c_old = p_c
            d_c_old = d_c

        elif len(x_c) == 1:
            missed_flag = True
            if np.sum((np.array([[x_c[0]], [y_c[0]]]) - p_c_old) ** 2) < \
                    np.sum((np.array([[x_c[0]], [y_c[0]]]) - d_c_old) ** 2):
                missed_marker_light = 'd'
                p_c = np.array([[x_c[0]], [y_c[0]]])
                d_c = np.array([[np.nan], [np.nan]])
                p_c_old = p_c

            else:
                missed_marker_light = 'p'
                d_c = np.array([[x_c[0]], [y_c[0]]])
                p_c = np.array([[np.nan], [np.nan]])
                d_c_old = d_c
        else:
            missed_flag = True
            d_c = np.array([[np.nan], [np.nan]])
            p_c = np.array([[np.nan], [np.nan]])

        Results.loc[2*INDEX, "X"] = p_c[0]
        Results.loc[2*INDEX+1, "X"] = d_c[0]

        Results.loc[2*INDEX, "Y"] = p_c[1]
        Results.loc[2*INDEX+1, "Y"] = d_c[1]

    Results = Results.dropna()
    Results.to_csv(PATH_TO_RESULTS, index=True)