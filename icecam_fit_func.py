"""
All functions needed for fitting
"""
import icecam_basic as cam
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
#Basic functions without distortion
def PRTX(points,alpha,beta,gamma,x,y,z):
    """
    Translates and then rotates around origin, returning pin projection.
    """
    points = cam.pinhole(cam.R(cam.T(points,x,y,z),alpha,beta,gamma))
    return points

def ravel_PRTX(points,alpha,beta,gamma,x,y,z):
    """
    Translates and then rotates around origin, returning pin projection,
    points get raveled at the end so it can be used for fits.
    """

    points = cam.pinhole(cam.R(cam.T(points,x,y,z),alpha,beta,gamma))

    return np.ravel(points[:2,:])

#Distortion function
def D(points,k1,k2,k3,k4):
    """
    Maps the distortion function. Returns points in distortion space.
    """
    x, y     = points[0,:], points[1,:]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)

    theta_d = theta*(1+k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)

    x = (theta_d/r)*x
    y = (theta_d/r)*y
    z = np.ones(x.shape)

    return np.vstack((x,y,z))

def equisolid_D(points):
    x, y     = points[0,:], points[1,:]
    r = np.sqrt(x**2 + y**2)

    theta_d = 2*np.sin(np.arctan(r)/2)

    x = (theta_d/r)*x
    y = (theta_d/r)*y
    z = np.ones(x.shape)

    return np.vstack((x,y,z))

#Complete fit function
def fit_func(world_points, fx, fy, cx, cy, k1, k2, k3, k4, *argv):
    """
    The function to be fitted with scipy curvefit. Input is a bit fucked here:
    Input is a 4xN array, with [0,1,2]xN being the world coordinates of each
    point, and [3]xN the number of the image they belong to.
    """
    for i in range(int(max(world_points[-1,:])+1)):
        o_points = world_points[0:3,np.where(world_points[-1,:]==i)[0]]
        pin_points = PRTX(o_points,*argv[i*6:(i+1)*6])
        fin_points = cam.K(D(pin_points,k1,k2,k3,k4),fx,fy,cx,cy)
        if i ==0:
            rep_points = fin_points
        else:
            rep_points = np.concatenate((rep_points,fin_points),axis=1)

    return np.asarray(rep_points).ravel()

def poE_func(world_points,alpha,beta,gamma,x,y,z,cx,cy):
    """
    Function to be fitted with scipy curvefit for the purpose of pose_estimation
    Does not need the same functionality as fit_func, because only one image is
    fitted at a time.
    """
    pin_points = PRTX(world_points,alpha,beta,gamma,x,y,z)
    fin_points = cam.K(equisolid_D(pin_points),-480,-480,cx,cy)

    return fin_points.ravel()


def initial_guess(objpoints,imgpoints): #,shape)
    """
    Providing initial guesses for each image based on the center part of image,
    which is assumed to be linear.
    """
    #p0=[0,0,0,0,0,0.1]
    #p0=[0,0,0,10,10,10]
    p0 = [0,0,0,0,0,0.6]
    #Return fit parameter as initial guess
    param_guess = so.curve_fit(ravel_PRTX, objpoints, np.ravel(imgpoints),p0)

    return param_guess


def img_obj_fit(imgpoints, objpoints):
    """
    return img_fit, obj_fit
    """
    img_fit = np.concatenate(imgpoints, axis=1)

    for i, xy_plane in enumerate(objpoints):
        if i == 0:
            obj_fit = np.vstack((np.array(xy_plane), np.ones(xy_plane.shape[1]) * i))
        else:
            num_array = np.vstack((np.array(xy_plane), np.ones(xy_plane.shape[1]) * i))
            obj_fit = np.concatenate((obj_fit, num_array), axis=1)


    return img_fit, obj_fit