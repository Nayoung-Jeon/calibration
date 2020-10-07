"""
Author : Gerrit Roellinghoff (SKKU)October 2019
Purpose: Basic operations for the arducam .RAW images. Including:
	 -Reading RAW images
	 -Bayer-Conversion
	 -Saving images in different formats
"""
#General Imports for calculation purposes
import glob
import numpy_unit as npu
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#Image libraries
import cv2

#Utilities

def read_raw(filename, height=979, width=1312, bayer = False):

    """Reads in .RAW file, returns as 2D numpy array with correct dimensions."""

    raw_file =  open(filename,'rb')
    image = (np.fromfile(raw_file, count = height*width, dtype='uint16'))/256
    image = np.reshape(image, (height,width), 'C')

    if bayer == True:
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

    return image.astype('uint8')


#read_raw("D:/icecube/3dboxdot/biggerdot/example/d0000922_3_Loop0.RAW")


def cvt_raw_ndarr(filename):
    horiz_size, vert_size= 1312, 979
    raw = np.fromfile(filename, dtype=np.uint16, count=horiz_size * vert_size) >> 4  # for little-endian type data
    raw = np.reshape(raw, (vert_size, horiz_size), 'C')
    #if bayer == True:
    image = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)

    image = image[14:]
    fig, ax = plt.subplots(1, figsize=(15, 10))
    plt.imshow(raw, cmap=plt.get_cmap('gray'))#, interpolation='none')
    clb = plt.colorbar()
    #clb.ax.set_ylabel('Pixel Count')
    #plt.savefig(filename+".png")
    print(raw.shape)
    raw=np.array(raw)
    print("haha")
    return image.astype(np.uint8)



def read_jpg(datadir):
    """
    Reads in all images in a folder, returns them as an array
"""
    images_path = np.sort(np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".jpg") ]))
    images = [cv2.imread(individual_path) for individual_path in images_path]

    return images

def read_png(datadir):
    """
    Reads in all images in a folder, returns them as an array
    """
    images_path = np.sort(np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ]))
    images = [cv2.imread(individual_path) for individual_path in images_path]

    return images


def save_img(filename, img):
    """
    Saves numpy array to image
    """
    cv2.imwrite(filename+".png",img)

def convert_folder(datadir,target):
    """
    Converts an entire folder of RAW files to png
    """
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".RAW") ])

    for img in images:
        png = read_raw(img)
        save_img(target+img.split("/")[-1].split(".")[0], png)
"""
def recursive_convert(datadir):
	
	//Converts all RAW files in folder and all of its subfolders to png.
	
    dirlist = [x[0] for x in os.walk(datadir)]
    print(dirlist)
    
    for sdir in dirlist:
        images = np.array([sdir +"/"+ f for f in os.listdir(sdir) if f.endswith(".RAW") ])

        for img in images:
            try:
                png = read_raw(img)
                save_img(sdir+"/"+img.split("/")[-1].split(".")[0], png)
            except:
                pass

"""
#Create ChAruco boards
#Transformation
##Rotation:
def R(points,alpha,beta,gamma):
    """
    Rotates points relative to world coordinates point of origin (lense).
    Camera always looks in positive z direction
    """
    Rx = np.array([[1,0,0],
                   [0,np.cos(alpha),-np.sin(alpha)],
                    [0,np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma), np.cos(gamma),0],
                   [0,0,1]])

    return np.dot(np.dot(Rz,np.dot(Ry,Rx)),points)

##Translation
def T(points,x,y,z):
    """
    Translate points according to world coordinates point of origin. Camera
    always looks in positive z direction.
    """
    return points + np.array([[x],[y],[z]])

"""
# read_jpg
def read_jpg(datadir):
    
    insert datadir = 'datapath/*.jpg'
    
    images_path = glob.glob(datadir)
    images = []

    for i in images_path:
        image = cv2.imread(i)
        images.append(image)

    return images
"""


##Pinhole
def pinhole(points):
    """
    Calculates the pinhole projection for a 3d image through origin (0,0,0)
    focal length is 1 unit.Returned is the virtual pinhole image in direction
    of camera.
    """
    return points/points[2,:]

##Camera projection
def K(points,fx,fy,cx,cy):
    """
    Calculates img coordinates based on virtual pinhole/distorted world points.
    """
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    return np.dot(K,points)[:2,:]


