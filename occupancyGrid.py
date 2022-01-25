import os
import python_data_fusion as pdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FILE_NAME = '10-Aug-2020-16-29'
# FILE_NAME = '20-Nov-2019-11-16'
# FILE_NAME = "04-Feb-2020-12-45"

CWD_PATH = os.getcwd()
PATH_TO_CSV = os.path.join(CWD_PATH, FILE_NAME, 'RADAR-lined-' + FILE_NAME + '.csv')

RESULT_DIR = os.path.join(CWD_PATH, FILE_NAME, "radar heat plot")
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

Data = pd.read_csv(PATH_TO_CSV, sep=',')

n = 90

radar = np.zeros((n, n, 100))

for INDEX in range(Data.__len__()):
    x_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR X Passenger"])).reshape(-1)
    y_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Y Passenger"])).reshape(-1)

    x_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR X Driver"])).reshape(-1)
    y_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Y Driver"])).reshape(-1)

    x_r = np.concatenate([x_d, x_p], axis=0)
    y_r = np.concatenate([y_d, y_p], axis=0)

    val_p = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Val Passenger"])).reshape(-1)
    val_d = np.array(pdf.csvCellReader(Data.loc[INDEX, "RADAR Val Driver"])).reshape(-1)

    z_r = np.concatenate([val_d, val_p], axis=0)

    x_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera X"]))
    y_c = np.array(pdf.csvCellReader(Data.loc[INDEX, "Camera Y"]))

    y_c = y_c[x_c.argsort()]
    x_c.sort()

    try:
        x_c = np.array([[x_c[0]], [x_c[19]]])
        y_c = np.array([[y_c[0]], [y_c[19]]])
    except:
        x_c = np.nan
        y_c = np.nan

    x, y, radar = pdf.heat_map_polar(radar, x_r, y_r, z_r, [0, 3], [-np.pi/2, np.pi/2], rbinnum=n, phibinnum=n)
    # x, y, radar = pdf.heat_map(radar, x_r, y_r, z_r, [-2, 2], [0, 5], xc=x_c, yc=y_c, xbinnum=n, ybinnum=n)



    if True:
        im = plt.pcolormesh(y, x, np.sum(radar.T, axis=0).T)
        plt.colorbar(im)
        plt.title("Epoch %d" %INDEX)
        plt.xlabel("phi")
        plt.ylabel("r(m)")

        FIG_NAME = "radar_epoch_%d.jpeg" %INDEX
        plt.savefig(os.path.join(RESULT_DIR, FIG_NAME), format='jpeg')

    plt.show(block=False)
    plt.pause(0.03)
    plt.close()