"""
Created on Fri Oct 25 11:30:38 2019
Purpose: Classical fisheye checkerboard calibration using OpenCV
@author: roellinghoff
"""
#Set environment
import sys
import icecam_fit_func as ff
import os
import json
import datetime
import icecam_points as p
import numpy as np
import random
import cv2
import scipy.optimize as so
import matplotlib.pyplot as plt
import icecam_basic as cam


#Create the output for this program
def output(images, i_points, r_points, result, save_folder_path):
    """
    Output of the program:
        - Reprojection plot for each image.
        - Reprojection error for each image
        - Reprojection error complete
        - Camera parameters as json file
    """
    savepath = save_folder_path+"/dot_{date:%Y%m%d_%H%M%S}/".format(date=datetime.datetime.now())
    os.mkdir(savepath)

    res = result[0]
    err = np.sqrt(np.diag(result[1]))

    for i,img in enumerate(images):
        try:
            os.mkdir(savepath+"image_{}".format(i))
        except:
            print(savepath+"image_{} already exists".format(i))

        plt.figure()
        plt.plot(i_points[i][0,:],i_points[i][1,:],"o",color = "#dae772", label="Image points")
        plt.plot(r_points[i][0,:],r_points[i][1,:],"rx", label =  "Fitted World points")
        plt.title("Image fit")
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.05), ncol=2)

        plt.savefig(savepath + "image_{}/fit_visualization.png".format(i))
        plt.close()

        error = i_points[i]-r_points[i]
        plt.figure()
        print("error:", error)
        plt.plot(error[0,:],error[1,:],"bo")
        plt.title("Reprojection residuals")
        plt.xlabel("Pixel")
        plt.ylabel("Pixel")

        plt.savefig(savepath + "image_{}/rep_error.png".format(i))
        plt.close()

        cv2.imwrite(savepath + "image_{}/image.png".format(i),img)

    i_pm = {"K":{"fx":res[0],"fy":res[1],"cx":res[2],"cy":res[3]},
            "K_err":{"fx":err[0],"fy":err[1],"cx":err[2],"cy":err[3]},
            "D":[res[4],res[5],res[6],res[7]],
            "D_err":[err[4],err[5],err[6],err[7]]}

    e_pm = {}
    n = 180/np.pi
    for i in range(len(images)):
        R = {"alpha":res[8+6*i]*n,"beta":res[9+6*i]*n,"gamma":res[10+6*i]*n}
        R_err = {"alpha":err[8+6*i]*n,"beta":err[9+6*i]*n,"gamma":err[10+6*i]*n}
        T = {"X":res[11+6*i],"Y":res[12+6*i],"Z":res[13+6*i]}
        T_err = {"X":err[11+6*i],"Y":err[12+6*i],"Z":err[13+6*i]}

        e_pm["image_{}".format(i)] = {"R":R,"R_err":R_err,
                                      "T":T,"T_err":T_err}

    data = {"intrinsic parameter":i_pm, "extrinsic_parameter":e_pm}

    with open(savepath+"result.json", "w") as outfile:
        json.dump(data, outfile, indent=4,sort_keys=True)

        np.save(savepath+"covariance.npy",result[1])
    return np.mean(np.sqrt(error[0,:]**2+error[1,:]**2))

#Main
def main(datadir):
    #Get all images for calibration
    images = cam.read_png(datadir)
    #Define amount of corners in each image
    size = (18,13)
    #Use Checkerboard properties to find points
    objpoints,imgpoints = p.checker_img_points(images, p.obj_points(size,20.0), size)
    #Determine initial parameters p0 by fitting linearly
    p_ext = []
    for i in range(len(objpoints)):
        fit = ff.initial_guess(objpoints[i],imgpoints[i])
        p_ext = p_ext + list(fit[0])
    p0 = [500,500,675,320,0,0,0,0] + p_ext

    #Convert both world and img points to format that can be used by fit
    img_fit   = np.concatenate(imgpoints,axis=1)
    for i,plane in enumerate(objpoints):
        if i==0:
            obj_fit = np.vstack((np.array(plane),np.ones(plane.shape[1])*i))
        else:
            num_array = np.vstack((np.array(plane),np.ones(plane.shape[1])*i))
            obj_fit = np.concatenate((obj_fit,num_array), axis=1)

    #Do the curve fit
    res = so.curve_fit(ff.fit_func,obj_fit,img_fit.ravel(),p0)
    #Reproject points with resulting parameters
    rep_points= np.reshape(ff.fit_func(obj_fit,*res[0]),img_fit.shape)

    #Split back into corresponding images
    img_length = np.cumsum([x.shape[1] for x in imgpoints])
    i_points = np.split(img_fit,img_length[:-1],axis=1)
    r_points = np.split(rep_points,img_length[:-1],axis=1)

    #Create plots and output json
    rep_error = output(images,i_points,r_points, res)

    print(rep_error)
    return rep_error






"""
#Execute
if __name__ == "__main__":
    main()
    if len(sys.argv) == 4:
        main()
    elif sys.argv[1] == "help":
        print("Usage: 'script' 'filefolder' 'checkersize' 'square_length'")
    else:
        print("Usage: 'script' 'filefolder' 'checkersize' 'square_length'")
"""