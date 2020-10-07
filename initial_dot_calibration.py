import cv2
from PIL import Image
import Fisheye_Calibration
import icecam_points as p
import icecam_fit_func as ff
import icecam_basic as cam
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import detectingblob as dtblob
import scipy.optimize as so

def dot_main(datadir, size, detector, savepath):
    images, objpoints, objpts, imgpoints, gray_list = dtblob.findcirclegrid_objpts_imgpts(datadir, detector, size)

#    print(objpts)
 #   print(imgpoints)
  #  print(gray_list)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpoints, gray_list[0].shape[::-1], None, None)
    print("ret", ret)
    print("mtx", mtx)
    print("dist", dist)
    print("rvecs", rvecs)
    print("tvecs", tvecs)

    return ret, mtx, dist, rvecs, tvecs

size=(17,27)
datadir="C:/Users/skdud/Downloads/dot file/real_test_dot"
detector=dtblob.sampleblobdetector
savepath="C:/Users/skdud/Downloads/dot file/real_test_dot/result_dot"

ret, mtx, dist, rvecs, tvecs = dot_main(datadir, size, detector, savepath)

"""==this is the result value==
ret = 0.4328513203102746
mtx =  [[1.47704926e+03, 0.00000000e+00, 7.46269690e+02],
 [0.00000000e+00, 1.47391777e+03, 1.00801946e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist = [[ 2.32428603e-01, -9.72237572e-01, -1.53670397e-04,  1.84388877e-03,
   1.30366097e+00]]

rvecs = [np.array([[ 0.05097275],
       [-0.21708515],
       [-0.00620109]])]
tvecs = [np.array([[-137.55023414],
       [-250.0734357 ],
       [ 455.9398206 ]])]
"""