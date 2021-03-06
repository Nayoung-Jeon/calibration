import cv2
import numpy as np
import matplotlib.pyplot as plt
import icecam_points as p
import icecam_basic as cam
from PIL import Image
import glob
import tagclick as tck
#import detectingblob as dtblob





# == Parameters for detecting both circles ==
bigdotparams = cv2.SimpleBlobDetector_Params()

## Change thresholds
bigdotparams.minThreshold = 0
bigdotparams.maxThreshold = 1000

## Filter by Area
bigdotparams.filterByArea = True
bigdotparams.minArea = 300
bigdotparams.maxArea = 10000

## Filter by Circularity
bigdotparams.filterByCircularity = True
bigdotparams.minCircularity = 0.1

## Filter by Convexity
bigdotparams.filterByConvexity = True
bigdotparams.minConvexity = 0.2

## Filter by Inertia
bigdotparams.filterByInertia = True
bigdotparams.minInertiaRatio = 0.3

# Create a detector with the parameters
boxbigdotblobdetector = cv2.SimpleBlobDetector_create(bigdotparams)





# == Parameters for detecting both circles ==
boxdotparams = cv2.SimpleBlobDetector_Params()

## Change thresholds
boxdotparams.minThreshold = 10
boxdotparams.maxThreshold = 200

## Filter by Area
boxdotparams.filterByArea = True
boxdotparams.minArea = 100
boxdotparams.maxArea = 100000

## Filter by Circularity
boxdotparams.filterByCircularity = True
boxdotparams.minCircularity = 0.6

## Filter by Convexity
boxdotparams.filterByConvexity = True
boxdotparams.minConvexity = 0.7

## Filter by Inertia
boxdotparams.filterByInertia = True
boxdotparams.minInertiaRatio = 0.3

# Create a detector with the parameters
boxdotblobdetector = cv2.SimpleBlobDetector_create(boxdotparams)






# == Parameters for detecting both circles ==
bothparams = cv2.SimpleBlobDetector_Params()

## Change thresholds
bothparams.minThreshold = 10
bothparams.maxThreshold = 200

## Filter by Area
bothparams.filterByArea = True
bothparams.minArea = 200
bothparams.maxArea = 100000

## Filter by Circularity
bothparams.filterByCircularity = True
bothparams.minCircularity = 0.7

## Filter by Convexity
bothparams.filterByConvexity = True
bothparams.minConvexity = 0.5

## Filter by Inertia
bothparams.filterByInertia = True
bothparams.minInertiaRatio = 0.2

# Create a detector with the parameters
bothblobdetector = cv2.SimpleBlobDetector_create(bothparams)


# == Parameters for detecting both circles ==
sampleparams = cv2.SimpleBlobDetector_Params()

## Change thresholds
sampleparams.minThreshold = 10
sampleparams.maxThreshold = 200

## Filter by Area
sampleparams.filterByArea = True
sampleparams.minArea = 70
sampleparams.maxArea = 100000

## Filter by Circularity
sampleparams.filterByCircularity = True
sampleparams.minCircularity = 0.7

## Filter by Convexity
sampleparams.filterByConvexity = True
sampleparams.minConvexity = 0.5

## Filter by Inertia
sampleparams.filterByInertia = True
sampleparams.minInertiaRatio = 0.1

# Create a detector with the parameters
sampleblobdetector = cv2.SimpleBlobDetector_create(sampleparams)



def RAW_findcirclegrid_3d(datadir, detector, size):
    """
    size = pattern grid (x, y, z) :
    savepath ="savepath/".format( date=datetime.datetime.now())
    imagepixelsize = pixel shape (n, m) = gray.shape

    return objpoints, imgpoints
    """

    raws_path = glob.glob(datadir + "/*.RAW")
    images = []

    for i, raw_path in enumerate(raws_path):
        img = cam.read_raw(raw_path)
        images.append(img)
        print(raw_path)

    right_objpoint = p.right_obj_points(size, 20)
    left_objpoint = p.left_obj_points(size, 20)

    right_objpoints = []
    right_imgpoints = []
    left_objpoints = []
    left_imgpoints = []
    tot_objpoints = []
    tot_imgpoints = []

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print("image number:", i)

        x_0, y_0 = tck.click_origin(img)[-1]

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print(img.shape)

        # find the centers of the circles in the grid
#         ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector)
#        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector)


        #if you use bigger dot, use this one
        ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector, flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))
        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector, flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

        # Cropping the image, so adding x_0 makes it same as the original image pixel

        right_corners.T[0][0] += x_0

        left_chess_img = cv2.drawChessboardCorners(img, (size[2], size[1]), left_corners, ret1)
        right_chess_img = cv2.drawChessboardCorners(img, (size[0], size[1]), right_corners, ret2)

        cv2.imshow('draw corner', left_chess_img)
        cv2.imshow('draw corner', right_chess_img)
        cv2.waitKey(2000)

        if ret1 == True:
            left_imgpoints.append((left_corners.T[:, 0, :]).astype('float32'))

            left_objpoints.append((left_objpoint.T).astype(int))

        if ret2 == True:
            # corn_res = corners.reshape(-1, 2).astype('float32')
            right_imgpoints.append((right_corners.T[:, 0, :]).astype('float32'))
            right_objpoints.append((right_objpoint.T).astype(int))
            # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # make left and right list to total list (imgpoints, objpoints)
        x_tot = np.concatenate((left_imgpoints[i][0], right_imgpoints[i][0]))
        x_tot = x_tot.tolist()
#        y_tot = np.concatenate((left_imgpoints[i][1]*(-1)+img.shape[0], right_imgpoints[i][1]*(-1)+img.shape[0]))
        y_tot = np.concatenate((left_imgpoints[i][1], right_imgpoints[i][1]))
        y_tot = y_tot.tolist()

        pre_tot_imgpoints = np.zeros((2, len(x_tot)))
        pre_tot_imgpoints[0], pre_tot_imgpoints[1] = x_tot, y_tot
        tot_imgpoints.append(pre_tot_imgpoints)

        pre_tot_objpoints = np.vstack([left_objpoints[i].T, right_objpoints[i].T]).T
        tot_objpoints.append(pre_tot_objpoints)

        # try imagepoints graph
        #img_fit = np.concatenate(tot_imgpoints, axis=1)
        #img_length = np.cumsum([x.shape[1] for x in tot_imgpoints])
        #i_points = np.split(img_fit, img_length[::-1], axis=1)

        # print(tot_objpoints)

        #plt.plot(i_points[i][0, :][100:], i_points[i][1, :][100:], "o", color="#dae772", label="Image points")
        #plt.plot(tot_objpoints[i][0][100:], tot_objpoints[i][1][100:], "rx", label="Fitted World points")

        #plt.show()

    return images, tot_imgpoints, tot_objpoints




def RAW_findcirclegrid_3d_right_cut(datadir, detector, size):
    """
    size = pattern grid (x, y, z) :
    savepath ="savepath/".format( date=datetime.datetime.now())
    imagepixelsize = pixel shape (n, m) = gray.shape

    return objpoints, imgpoints
    """

    raws_path = glob.glob(datadir + "/*.RAW")
    images = []

    for i, raw_path in enumerate(raws_path):
        img = cam.read_raw(raw_path)
        images.append(img)
        print(raw_path)

    right_objpoint = p.right_obj_points(size, 20)
    left_objpoint = p.left_obj_points(size, 20)

    right_objpoints = []
    right_imgpoints = []
    left_objpoints = []
    left_imgpoints = []
    tot_objpoints = []
    tot_imgpoints = []

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print("image number:", i)

        x_0, y_0 = tck.click_origin(img)[-1]

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print(img.shape)

        # find the centers of the circles in the grid
#         ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector)
#        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector)


        #if you use bigger dot, use this one
        ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector, flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))
        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector, flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

        # Cropping the image, so adding x_0 makes it same as the original image pixel

        right_corners.T[0][0] += x_0

        left_chess_img = cv2.drawChessboardCorners(img, (size[2], size[1]), left_corners, ret1)
        right_chess_img = cv2.drawChessboardCorners(img, (size[0], size[1]), right_corners, ret2)

        cv2.imshow('draw corner', left_chess_img)
        cv2.imshow('draw corner', right_chess_img)
        cv2.waitKey(200)

        if ret1 == True:
            left_imgpoints.append((left_corners.T[:, 0, :]).astype('float32'))

            left_objpoints.append((left_objpoint.T).astype(int))

        if ret2 == True:
            # corn_res = corners.reshape(-1, 2).astype('float32')
            right_imgpoints.append((right_corners.T[:, 0, :]).astype('float32'))
            right_objpoints.append((right_objpoint.T).astype(int))
            # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # make left and right list to total list (imgpoints, objpoints)
        x_tot = np.concatenate((left_imgpoints[i][0], right_imgpoints[i][0][:10]))
        x_tot = x_tot.tolist()
#        y_tot = np.concatenate((left_imgpoints[i][1]*(-1)+img.shape[0], right_imgpoints[i][1]*(-1)+img.shape[0]))
        y_tot = np.concatenate((left_imgpoints[i][1], right_imgpoints[i][1][:10]))
        y_tot = y_tot.tolist()

        pre_tot_imgpoints = np.zeros((2, len(x_tot)))
        pre_tot_imgpoints[0], pre_tot_imgpoints[1] = x_tot, y_tot
        tot_imgpoints.append(pre_tot_imgpoints)

        pre_tot_objpoints = np.vstack([left_objpoints[i].T, right_objpoints[i].T[:10]]).T
        tot_objpoints.append(pre_tot_objpoints)

        # try imagepoints graph
        #img_fit = np.concatenate(tot_imgpoints, axis=1)
        #img_length = np.cumsum([x.shape[1] for x in tot_imgpoints])
        #i_points = np.split(img_fit, img_length[::-1], axis=1)

        # print(tot_objpoints)

        #plt.plot(i_points[i][0, :][100:], i_points[i][1, :][100:], "o", color="#dae772", label="Image points")
        #plt.plot(tot_objpoints[i][0][100:], tot_objpoints[i][1][100:], "rx", label="Fitted World points")

        #plt.show()

    return images, tot_imgpoints, tot_objpoints

