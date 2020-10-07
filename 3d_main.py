# https://www.iitr.ac.in/departments/MA/uploads/Unit%202%20lec-1.pdf 참고 ppt
# https://eyebug.wordpress.com/2010/10/13/camera-calibration/
# https://github.com/begumcig/Camera-Calibration/commit/974d84a1114d9aec3f9ed9690294392e963d1a74
# 카메라 3d에 대해 어떻게 변환시키는 지 슬라이드 https://slideplayer.com/slide/7596427/
# xyz축 찾는 법 https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
# real 3d calibrate https://github.com/bailus/tsai-calibration/blob/master/main.py
# 관련 논문 https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-801-machine-vision-fall-2004/readings/tsaiexplain.pdf
# how to find corners https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html

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
import detect3d as dtb3d
import scipy.optimize as so
import tagclick as tlk

"""
def dot_main(datadir, size, detector, savepath):

    images, objpoints, imgpoints = dtblob.findcirclegrid_objpts_imgpts(datadir, detector, size)

    p_ext=[]

    for i in range(len(objpoints)):
        fit = ff.initial_guess(objpoints[i], imgpoints[i])
        p_ext = p_ext+list(fit[0])

    p0 = [300,300,1000,1000,0,0,0,0] + p_ext

    img_fit, obj_fit = ff.img_obj_fit(imgpoints, objpoints)

    #curve_fit
    res = so.curve_fit(ff.fit_func, obj_fit, img_fit.ravel(), p0)

    #reproject object points
    rep_points = np.reshape(ff.fit_func(obj_fit, *res[0]), img_fit.shape)

    # Split back into corresponding images
    img_length = np.cumsum([x.shape[1] for x in imgpoints])
    i_points = np.split(img_fit, img_length[:-1], axis=1)
    r_points = np.split(rep_points, img_length[:-1], axis=1)

    # Create plots and output json
    rep_error = Fisheye_Calibration.output(images, i_points, r_points, res, savepath)
    print(rep_error)

    return rep_error
    
    
    
    for i in range(len(objpoints)):
        plt.plot(i_points[i][0, :], i_points[i][1, :], "o", color="#dae772", label="Image points")
        plt.plot(r_points[i][0, :], r_points[i][1, :], "rx", label="Fitted World points")
        plt.title("Image fit")
        plt.show()


size=(18,14)
datadir="C:/Users/skdud/Desktop/icecube/dot file/3d"
detector=dtblob.sampleblobdetector
savepath="C:/Users/skdud/Desktop/icecube/dot file/3d/result"
dot_main(datadir, size, detector, savepath)


"""


#Main
def dtmain(datadir, savepath, detector, size):
    #print("haha")

#    images,  imgpoints, objpoints = dtb3d.RAW_findcirclegrid_3d(datadir, detector, size)
    images,  imgpoints, objpoints = dtb3d.RAW_findcirclegrid_3d(datadir, detector, size)

    #print(imgpoints); print(objpoints)

    #Determine initial parameters p0 by fitting linearly
    p_ext = []

    for i in range(len(objpoints)):
        #print("obj", objpoints[i]); print("img", imgpoints[i])
        fit = ff.initial_guess(objpoints[i],imgpoints[i])
        p_ext = p_ext + list(fit[0])

    p0 = [800, 500, 600, 300, 0, 0, 0, 1] + p_ext
   # p0 = [900,500,600,300,0,0,0,1] + p_ext
  #  p0 = [800, 600, 700, 300, 0, 0, 0, 1] + p_ext

    #Convert both world and img points to format that can be used by fit
    img_fit   = np.concatenate(imgpoints,axis=1)
    for i,plane in enumerate(objpoints):
        if i==0:
            obj_fit = np.vstack((np.array(plane),np.ones(plane.shape[1])*i))
        else:
            num_array = np.vstack((np.array(plane),np.ones(plane.shape[1])*i))
            obj_fit = np.concatenate((obj_fit,num_array), axis=1)

    #print(img_fit)
    #print(obj_fit)

    #Do the curve fit
    res = so.curve_fit(ff.fit_func,obj_fit,img_fit.ravel(),p0)
    #Reproject points with resulting parameters
    rep_points= np.reshape(ff.fit_func(obj_fit,*res[0]),img_fit.shape)

    #Split back into corresponding images
    img_length = np.cumsum([x.shape[1] for x in imgpoints])
    i_points = np.split(img_fit,img_length[:-1],axis=1)
    r_points = np.split(rep_points,img_length[:-1],axis=1)

    for i in range(len(objpoints)):
        plt.plot(i_points[i][0, :], i_points[i][1, :], "o", color="#dae772", label="Image points")
        plt.plot(r_points[i][0, :], r_points[i][1, :], "rx", label="Fitted World points")
        plt.title("Image fit")
        plt.show()

    #Create plots and output json
    rep_error = Fisheye_Calibration.output(images, i_points, r_points, res, savepath)
    print(rep_error)

    return rep_error


datadir="D:/icecube/3dboxdot/biggerdot/example/d"
savepath="D:/icecube/3dboxdot/biggerdot/example/result"


filename="D:/icecube/3dboxdot/middle/d0000910_0_Loop0.RAW"
dotgridsize=(7,6,7)
dotgridsize2=(11,8,11)
#print(cam.cvt_raw_ndarr(filename))

datadir2="D:/icecube/3dboxdot/left/test"
savepath2="D:/icecube/3dboxdot/left/test/result"

#images,  imgpoints, objpoints = dtblob.findcirclegrid_objpts_imgpts_3d(datadir, detector, size)
dtmain(datadir, savepath, detector=dtb3d.boxbigdotblobdetector, size=dotgridsize)

#dtblob.makekeypoints(dtblob.boxdotblobdetector, cam.read_raw(filename))


#dtblob.RAW_findcirclegrid_objpts_imgpts_3d(datadir, dtblob.boxdotblobdetector, dotgridsize)