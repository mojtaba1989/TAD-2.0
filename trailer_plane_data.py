import glob
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import python_data_fusion as pdf
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append("..")

OUTPUT         = False
estimateCenter = False
trackEnable    = False
threshold      = False
REVISE         = False
MODEL_NAME     = 'inference_graph_FRCNN_V2'
UV             = False
count = 0
if __name__ == '__main__':
    for idx, arg in enumerate(sys.argv):
        if arg in ['--file', '-f']:
            FILE_NAME = str(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--graph', '-g']:
            OUTPUT = 'graph'
            del sys.argv[idx]
        elif arg in ['--image', '-i']:
            OUTPUT = 'image'
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--estimateCenter', '-ec']:
            estimateCenter = True
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--enable', '-e']:
            trackEnable = True
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--threshold', '-t']:
            threshold = float(sys.argv[idx + 1])
            del sys.argv[idx]
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--revise', '-r']:
            REVISE = True
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['-uv']:
            UV = True
            del sys.argv[idx]

    for idx, arg in enumerate(sys.argv):
        if arg in ['--model', '-m']:
            if str(sys.argv[idx + 1]) in ['ssd', 'SSD']:
                MODEL_NAME = 'inference_graph_SSDMN'
            else:
                MODEL_NAME = 'inference_graph_FRCNN_V2'
            del sys.argv[idx]
            del sys.argv[idx]

    CWD_PATH = os.getcwd()
    PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '.csv')
    PATH_TO_TXT = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '_Report.txt')
    PATH_TO_STXT = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '_Srores_Report.txt')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # making output csv
    dummy = pd.read_csv(PATH_TO_CSV)
    datatNum = len(dummy)
    if trackEnable:
        PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '-' + MODEL_NAME[16:] + '-Tracking' + '.csv')
    else:
        PATH_TO_RESULTS = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '-' + MODEL_NAME[16:] + '.csv')
    Results = pd.DataFrame(np.zeros([datatNum, 1]), columns=["Vernier"])
    XY = pd.DataFrame(np.zeros([datatNum, 1]))
    if UV:
        uv = pd.DataFrame(np.zeros([datatNum, 1]))
        if trackEnable:
            PATH_TO_UV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '-UV-tracking.csv')
        else:
            PATH_TO_UV = os.path.join(CWD_PATH, FILE_NAME, FILE_NAME + '-UV.csv')

    NUM_CLASSES = 3
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Object detection
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Marker-light tracking
    TRACK_FLAG = False
    multitracker = cv2.MultiTracker_create()


    # Initialization

    cdata = np.ones([10]) * 10

    cam_l_corner = np.nan
    cam_w_trailer = np.nan
    cam_var = 1

    p_c_old = np.nan
    d_c_old = np.nan
    l_p = np.empty((1, 1))


    l_d = np.empty((1, 1))

    var_t = .01


    center = [[-0.007], [.232]]

    # Main Loop
    g = open(PATH_TO_STXT, 'w')
    for INDEX in range(datatNum):
        index, time_stamp, angle = pdf.readCSV_CAM(PATH_TO_CSV, INDEX+1)
        Results.loc[INDEX, "Vernier"] = np.array(angle)

        IMG_NAME = "img_%d.jpeg" % index
        PATH_TO_IMAGE = os.path.join(CWD_PATH, FILE_NAME, 'Figures/', IMG_NAME)
        image = cv2.imread(PATH_TO_IMAGE)
        # image = pdf.img_filter(image, u_lim=[0, 1], v_lim=[.7, 1])

        tstart = time.time()
        if not trackEnable or not TRACK_FLAG:
            count += 1
            image_expanded = np.expand_dims(image, axis=0)
            (boxes_ml, scores_ml, classes_ml, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            scoreTXT = np.round(scores_ml, 3).reshape(-1)
            scoreTXT = scoreTXT[scoreTXT > 0]
            g.write(str(scoreTXT)[1:-1]+' ')
            boxes_ml = boxes_ml.reshape(-1, 4)
            if threshold is not False:
                boxes_ml = boxes_ml[(scores_ml > threshold).reshape(-1), ]
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='tf')
            for bbox in boxes_ml:
                multitracker.add(cv2.TrackerMedianFlow_create(), image, bbox)
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')
            if boxes_ml.shape[0] <= 1:
                TRACK_FLAG = False
                multitracker = cv2.MultiTracker_create()
            else:
                TRACK_FLAG = True

        else:
            success, boxes_ml = multitracker.update(image)
            boxes_ml = pdf.tfcv_convertor(boxes_ml, image.shape[0:2], source='cv')
            if boxes_ml.shape[0] <= 1:
                TRACK_FLAG = False
                multitracker = cv2.MultiTracker_create()
                count += 1
            else:
                TRACK_FLAG = True

        if UV:
            box_temp = pdf.de_normalize(pdf.mid_detection(boxes_ml), image.shape[:-1])
            u_temp = box_temp[:, 1]
            v_temp = box_temp[:, 0]
            v_temp = v_temp[u_temp.argsort()]
            u_temp.sort()
            try:
                uv.loc[2 * INDEX, "Ud"]     = u_temp[0]
                uv.loc[2 * INDEX, "Vd"]     = v_temp[0]
                uv.loc[2 * INDEX + 1, "Vd"] = v_temp[1]
                uv.loc[2 * INDEX + 1, "Ud"] = u_temp[1]
            except:
                pass
        mid = pdf.undistortPoint(pdf.mid_detection(boxes_ml), image.shape[:-1])
        # mid = pdf.undistortPoint(pdf.houghCenter(image, boxes_ml), image.shape[:-1])
        mid[mid == 0] = np.nan
        mid = mid[~np.isnan(mid).any(axis=1)]

        cam_det = pdf.reversePinHole(mid, [0, 0, .78], -9, 0, image.shape[:-1])
        # cam_det = pdf.img2radar_map(mid, .78, -6, 2.1, image.shape[:-1])

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
            if np.sum((np.array([[x_c[0]], [y_c[0]]])-p_c_old)**2) < \
                    np.sum((np.array([[x_c[0]], [y_c[0]]])-d_c_old)**2):
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



        if not estimateCenter:
            R = np.concatenate([p_c, d_c], axis=1) - center

            cam_phi, cam_l_corner, cam_w_trailer = pdf.update_measure(
                R, cam_l_corner, cam_w_trailer, cam_var, var_t)

            cam_var, cdata = pdf.res_var(cdata, cam_l_corner, n=10)
            tend = time.time()
            Results.loc[INDEX, "Cam"] = cam_phi
            Results.loc[INDEX, "L"] = cam_l_corner
            Results.loc[INDEX, "Lambda"] = cam_w_trailer
            Results.loc[INDEX, "Lp"] = np.sqrt(np.sum(R[:, 0] ** 2))
            Results.loc[INDEX, "Ld"] = np.sqrt(np.sum(R[:, 1] ** 2))
            Results.loc[INDEX, "Time"] = tend - tstart

        else:
            tend = time.time()
            Results.loc[INDEX, "Time"] = tend - tstart
            XY.loc[INDEX, "X_p"] = p_c[0]
            XY.loc[INDEX, "X_d"] = d_c[0]
            XY.loc[INDEX, "Y_p"] = p_c[1]
            XY.loc[INDEX, "Y_d"] = d_c[1]

        if OUTPUT == 'image':
            image = pdf.undistort_org(image)
            text = 'FPS:%d' % int(1 / Results.loc[INDEX, "Time"])
            image = cv2.putText(image, text, (1100, 900), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

            vis_util.draw_keypoints_on_image_array(image,
                                                   mid,
                                                   color='blue',
                                                   radius=5,
                                                   use_normalized_coordinates=True)
            cv2.imshow('Object detector', cv2.resize(image, (800, 600)))
            # cv2.imwrite('d'+IMG_NAME, image)
            cv2.waitKey(1)
        else:
            print(INDEX)

    if estimateCenter:
        x = np.hstack((np.array(XY["X_p"]), np.array(XY["X_d"])))
        y = np.hstack((np.array(XY["Y_p"]), np.array(XY["Y_d"])))
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        center, cam_l_corner, trailer_tilt, residual = pdf.centerCalc(x, y, initial_center=(0, .3, .1))
        cam_w_trailer = np.nanmean(np.sqrt((XY["X_p"]-XY["X_d"])**2 +
                                           (XY["Y_p"]-XY["Y_d"])**2))
        omega = 2*np.degrees(np.arcsin(cam_w_trailer/cam_l_corner/2))
        f = open(PATH_TO_TXT, 'w')
        f.write(('center ' + str(tuple(center))+'\n'))
        f.write(('l ' + str(cam_l_corner)+'\n'))
        f.write(('W ' + str(cam_w_trailer)+'\n'))
        f.write(('Omega ' + str(omega) + '\n'))
        f.write(('Trailer Tilt ' + str(trailer_tilt) + '\n'))
        f.write(('residual ' + str(residual)+'\n'))
        f.close()

        for INDEX in range(datatNum):
            p_c = np.empty((2, 1))
            d_c = np.empty((2, 1))
            p_c[0] = XY.loc[INDEX, "X_p"]
            d_c[0] = XY.loc[INDEX, "X_d"]
            p_c[1] = XY.loc[INDEX, "Y_p"]
            d_c[1] = XY.loc[INDEX, "Y_d"]

            R = np.concatenate([p_c, d_c], axis=1) - center

            cam_phi, cam_l_corner, cam_w_trailer = pdf.update_measure(
                R, cam_l_corner, cam_w_trailer, cam_var, np.nan)

            cam_var, cdata = pdf.res_var(cdata, cam_l_corner, n=10)

            Results.loc[INDEX, "Cam"] = cam_phi
            Results.loc[INDEX, "L"] = cam_l_corner
            Results.loc[INDEX, "Lambda"] = cam_w_trailer
            Results.loc[INDEX, "Lp"] = np.sqrt(np.sum(R[:, 0]**2))
            Results.loc[INDEX, "Ld"] = np.sqrt(np.sum(R[:, 1]**2))

    cv2.destroyAllWindows()
    g.close()

    if REVISE:
        from scipy.ndimage.interpolation import shift
        Results["Vernier"] = shift(Results["Vernier"], -20, cval=np.NaN)
    while True:
        try:
            Results.to_csv(PATH_TO_RESULTS, index=True)
            Results.to_csv("c:/TAD-Mojtaba/MATLAB/camera-" + FILE_NAME + '.csv', index=True)
            if UV:
                uv.to_csv(PATH_TO_UV, index=True)
            break
        except:
            print('Close that damn file idiot!!!')
            time.sleep(1)
    print(count)