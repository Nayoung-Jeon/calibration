"""
World and Image points identifier
"""
import cv2

import cv2
import numpy as np

def checker_img_points(images, objpoint, checkerboard):
    """
    Takes an array of checkerboard images, uses OpenCV
    "cv2.findChessboardCorners()" to find all chessboard corners and does
    subpixel refinement on them.
    """
    #Criteria for subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    objpoints = [] # 3d point in world space
    imgpoints = [] # 2d point in image plane

    dellist = []

    for i,img in enumerate(images):
        print("image number:", i)

        #when running raw_plane_chess, don't run below code
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = img

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objpoint.T)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            imgpoints.append(corners.T[:,0,:])
        else:
            print("Found no checkerboard in this image {}".format(i))
            cv2.imshow("image",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            dellist.append(i)

    for index in sorted(dellist, reverse=True):
        del images[index]

    print("Found {} checkerboards of size {}".format(len(objpoints),checkerboard))
    return objpoints,imgpoints

def charuco_img_points(images, objpoint, board, a_dict):
    """
    Takes an array of ChAruco images, uses OpenCv "cv2.aruco" to identify
    Aruco Corners and interpolate the chessboard corners, returns all
    identified corners and their corresponding world points.
    """
    #Criteria for subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    objpoints = [] # 3d point in world space
    imgpoints = [] # 2d point in image plane

    for img in images:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners, ids, rejpoints = cv2.aruco.detectMarkers(gray, a_dict)
        if len(corners)>0:
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None:
                cv2.cornerSubPix(gray,res2[1],(3,3),(-1,1),criteria)
                imgpoints.append(res2[1].T[:,0,:])
                objpoints.append(objpoint[:,res2[2].flatten()])
                cv2.aruco.drawDetectedCornersCharuco(img,res2[1],res2[2])
                cv2.imshow("frame",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return objpoints,imgpoints

def two_plane_obj_points(grid_size, dx):
    """
    Creates a 3D array of 3D world object points.
    If you want to calibrate 2-plane, you should make objectpoints as that.
    grid_size = (x_num, y_num, z_num)
    """
    objp_xy = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    objp_yz = np.zeros((grid_size[1]*grid_size[2], 3), np.float32)
    objp_xy[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp_yz[:,1:3] = np.mgrid[0:grid_size[1], 0:grid_size[2]].T.reshape(-1, 2)

    return objp_xy*dx, objp_yz*dx



def obj_points(grid_size, dx):
    """
    Creates a 3D array of 3D world object points. Z coordinate for every point
    is assumed to be zero
    Dimensions:
                1 x grid_size[0]*grid_size[1] x 3
    """
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    return objp*dx


def right_obj_points(grid_size, dx):
    """
    Creates a 3D array of 3D world object points. Z coordinate for every point
    is assumed to be zero
    Dimensions:
                1 x grid_size[0]*grid_size[1] x 3
    """
    objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[1:grid_size[0]+1, 0:grid_size[1]].T.reshape(-1, 2)

    return objp*dx



def left_obj_points(grid_size, dx):
    """
    Creates a 3D array of 3D world object points. X coordinate for every point
    is assumed to be zero
    Dimensions:
                1 x grid_size[0]*grid_size[1] x 3
    """
    objp = np.zeros((grid_size[1] * grid_size[2], 3), np.float32)
    objp[:,1:3] = np.mgrid[0:grid_size[2], 0:grid_size[1]].T.reshape(-1, 2)
    u=objp.T; y=0*u; y[1]=u[2]; y[2]=u[1]

    return y.T * dx




def cube_obj_points(dx):
    """
    Create a charuco obj points object to easily compare against found image
    points.
    """
    c = np.ones((1,8,9))*4.5
    sym   = np.concatenate((np.mgrid[0:8,0:9][::-1]-3.5,-c))
    asym  = np.concatenate((+c,np.mgrid[0:8,0:9][::+1]-3.5))
    sym2  = np.concatenate((-np.mgrid[0:8,0:9][::-1,::-1]+3.5,c))
    asym2 = np.concatenate((-c,-np.mgrid[0:8,0:9][::1,::-1]+3.5))

    box = np.concatenate((sym,asym,sym2,asym2),axis=2)[:,:,:-1]
    return box.reshape(3,-1)*dx

if __name__ == "__main__":
    print(cube_obj_points(1)[:,:36])