import cv2
import numpy as np
import matplotlib.pyplot as plt
import icecam_points as p
import icecam_basic as cam
from PIL import Image
import glob
import tagclick as tck

#set up blob detector


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




def resize_png(image_path, num):
    """
    large image cannot go on findCirclesGrid. So when you think the image size is big enough, try this.
    num is the number which you want to divide by.
    return gray and image
    if it was GRAY,
    """
    img=Image.open(image_path+".png")
    resize_img=img.resize((int(img.width/num), int(img.height/num)))
#    resize_img.save(image_path+"_cut.png")
#    img_bad=cv2.imread(image_path+"_cut.png")
    img_bad = np.array(resize_img)
    if len(img_bad.shape)==3:
        gray_bad=cv2.cvtColor(img_bad, cv2.COLOR_BGR2GRAY)
    else:
        gray_bad=img_bad
        print('already GRAY')

    return img_bad



def resize_jpg(image_path, num):
    """
    large image cannot go on findCirclesGrid. So when you think the image size is big enough, try this.
    num is the number which you want to divide by.
    """
    img=Image.open(image_path)
    resize_img=img.resize((int(img.width/num), int(img.height/num)))
#    resize_img.save(image_path+"_downpixel.jpg")
#    img_bad=cv2.imread(image_path+"_downpixel.jpg")
    img_bad=np.array(resize_img)

    return img_bad




def makekeypoints(detector, gray):
    """Via the detector you set, make keypoints and show them"""

    keypoints = detector.detect(gray)
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('imkey', im_with_keypoints)
    cv2.waitKey(5000)


    return keypoints


def keypoint2circle(keypoints, gray):
    center = []
    circle = []
    blackbackground = 255 * np.ones(gray.shape)

    for i, f in enumerate(keypoints):
        radius = int(f.size / 2)

        cx = f.pt[0]
        cy = f.pt[1]
        center.append([round(cx), round(cy)])
        circle.append(cv2.circle(blackbackground, (round(cx), round(cy)), 30, color=(0, 0, 0), thickness=-1))


    # draw plot
    center_x = np.array(center).T[0]
    center_y = np.array(center).T[1]
    plt.plot(center_x, center_y, 'o')
    plt.show()

    # imshow circle
    cv2.imshow("", circle[0])

    return center, circle[0]


def findcirclegrid_objpts_imgpts(datadir, detector, size) :
    """
    size = pattern grid (row, column) :
    savepath ="savepath/".format( date=datetime.datetime.now())
    imagepixelsize = pixel shape (n, m) = gray.shape

    return objpoints, imgpoints
    """
    images_path = glob.glob(datadir + "/*.jpg")
    images = []

    for i, img_path in enumerate(images_path):
        img = resize_jpg(img_path, num=2)
        images.append(img)

    objpoint = p.obj_points(size, 20)

    objpoints = []
    imgpoints = []


    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print ("image number:", i)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find the centers of the circles in the grid
        ret, corners = cv2.findCirclesGrid(gray, size, blobDetector=detector)
        chess_img = cv2.drawChessboardCorners(img, size, corners, ret)
        cv2.imshow('draw corner', chess_img)
        cv2.waitKey(5000)

        if ret == True:
            #corn_res = corners.reshape(-1, 2).astype('float32')
            imgpoints.append((corners.T[:,0,:]).astype('float32'))
            objpoints.append(objpoint.T)
            #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


    return images, objpoints, imgpoints



def findcirclegrid_objpts_imgpts_3d(datadir, detector, size) :
    
    #size = pattern grid (x, y, z) :
    #savepath ="savepath/".format( date=datetime.datetime.now())
    #imagepixelsize = pixel shape (n, m) = gray.shape

    #return objpoints, imgpoints


    images_path = glob.glob(datadir + "/*.png")
    images = []

    for i, img_path in enumerate(images_path):
        img = resize_jpg(img_path, num=2)
        images.append(img)

    right_objpoint = p.obj_points(size, 20)
    left_objpoint = p.left_obj_points(size, 20)

    right_objpoints = [];    right_imgpoints = []
    left_objpoints = [];    left_imgpoints =[]
    tot_objpoints = [];     tot_imgpoints=[]


    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print ("image number:", i)

        x_0, y_0 = tck.click_origin(img)[-1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find the centers of the circles in the grid
        ret1, left_corners = cv2.findCirclesGrid(gray.T[:x_0].T, (size[2], size[1]), blobDetector=detector)
        ret2, right_corners = cv2.findCirclesGrid(gray.T[x_0:].T, (size[0], size[1]), blobDetector=detector)

        # Cropping the image, so adding x_0 makes it same as the original image pixel
        right_corners.T[0][0] += x_0

        left_chess_img = cv2.drawChessboardCorners(img, (size[2], size[1]), left_corners, ret1)
        right_chess_img = cv2.drawChessboardCorners(img, (size[0], size[1]), right_corners, ret2)

        cv2.imshow('draw corner', left_chess_img)
        cv2.imshow('draw corner', right_chess_img)
        cv2.waitKey(500)

        if ret1 == True:
            left_imgpoints.append((left_corners.T[:,0,:]).astype('float32'))
            left_objpoints.append((left_objpoint.T).astype(int))


        if ret2 == True:
            #corn_res = corners.reshape(-1, 2).astype('float32')

            right_imgpoints.append((right_corners.T[:,0,:]).astype('float32'))
            right_objpoints.append((right_objpoint.T).astype(int))
            #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #make left and right list to total list (imgpoints, objpoints)
        le = np.concatenate((left_imgpoints[i][0], right_imgpoints[i][0])).tolist()
        ler = np.concatenate((left_imgpoints[i][1], right_imgpoints[i][1])).tolist()

        pre_tot_imgpoints = np.zeros((2, len(le)))
        pre_tot_imgpoints[0], pre_tot_imgpoints[1]= le, ler
        tot_imgpoints.append(pre_tot_imgpoints)

        pre_tot_objpoints = np.vstack([left_objpoints[i].T,right_objpoints[i].T]).T
        tot_objpoints.append(pre_tot_objpoints)

    return images,  tot_imgpoints, tot_objpoints



def RAW_findcirclegrid_objpts_imgpts_3d(datadir, detector, size) :
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

    right_objpoint = p.right_obj_points_test(size, 30)
    left_objpoint = p.left_obj_points_test(size, 30)

    right_objpoints = [];    right_imgpoints = []
    left_objpoints = [];    left_imgpoints =[]
    tot_objpoints = [];     tot_imgpoints=[]


    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print ("image number:", i)

        x_0, y_0 = tck.click_origin(img)[-1]

       # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(img.T[:x_0].T)

        #find the centers of the circles in the grid
        ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector)
        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector)

        # Cropping the image, so adding x_0 makes it same as the original image pixel
        right_corners.T[0][0] += x_0

        left_chess_img = cv2.drawChessboardCorners(img, (size[2], size[1]), left_corners, ret1)
        right_chess_img = cv2.drawChessboardCorners(img, (size[0], size[1]), right_corners, ret2)

        cv2.imshow('draw corner', left_chess_img)
        cv2.imshow('draw corner', right_chess_img)
        cv2.waitKey(200)



        if ret1 == True:
            left_imgpoints.append((left_corners.T[:,0,:]).astype('float32'))
            left_objpoints.append((left_objpoint.T).astype(int))


        if ret2 == True:
            #corn_res = corners.reshape(-1, 2).astype('float32')

            right_imgpoints.append((right_corners.T[:,0,:]).astype('float32'))
            right_objpoints.append((right_objpoint.T).astype(int))
            #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #make left and right list to total list (imgpoints, objpoints)



        x_img = np.concatenate((left_imgpoints[i][0], right_imgpoints[i][0])).tolist()
        y_img = np.concatenate((left_imgpoints[i][1], right_imgpoints[i][1]).tolist()

        pre_tot_imgpoints = np.zeros((2, len(x_img)))
        pre_tot_imgpoints[0], pre_tot_imgpoints[1]= x_img, y_img
        tot_imgpoints.append(pre_tot_imgpoints)

        pre_tot_objpoints = np.vstack([left_objpoints[i].T,right_objpoints[i].T]).T
        tot_objpoints.append(pre_tot_objpoints)

        # plot image points
        img_fit = np.concatenate(tot_imgpoints, axis=1)
        img_length = np.cumsum([x.shape[1] for x in tot_imgpoints])
        i_points = np.split(img_fit, img_length[:-1], axis=1)

        plt.plot(i_points[i][0, :], i_points[i][1, :], "o", color="#dae772", label="Image points")
        #plt.plot(tot_objpoints[i][2][:30], tot_objpoints[i][1][:30], "rx", label="Fitted World points")

        plt.show()

    return images,  left_imgpoints, left_objpoints


def RAW_findcirclegrid_objpts_imgpts_3_test(datadir, detector, size) :
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

    right_objpoint = p.right_obj_points_test(size, 20)
    left_objpoint = p.left_obj_points_test(size, 20)

    right_objpoints = [];    right_imgpoints = []
    left_objpoints = [];    left_imgpoints =[]
    tot_objpoints = [];     tot_imgpoints=[]


    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, img in enumerate(images):
        print ("image number:", i)

        x_0, y_0 = tck.click_origin(img)[-1]

       # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find the centers of the circles in the grid
        ret1, left_corners = cv2.findCirclesGrid(img.T[:x_0].T, (size[2], size[1]), blobDetector=detector)
        ret2, right_corners = cv2.findCirclesGrid(img.T[x_0:].T, (size[0], size[1]), blobDetector=detector)

        # Cropping the image, so adding x_0 makes it same as the original image pixel
        right_corners.T[0][0] += x_0

        left_chess_img = cv2.drawChessboardCorners(img, (size[2], size[1]), left_corners, ret1)
        right_chess_img = cv2.drawChessboardCorners(img, (size[0], size[1]), right_corners, ret2)

        cv2.imshow('draw corner', left_chess_img)
        cv2.imshow('draw corner', right_chess_img)
        cv2.waitKey(200)

        if ret1 == True:
            left_imgpoints.append((left_corners.T[:,0,:]).astype('float32'))
            left_objpoints.append((left_objpoint.T).astype(int))


        if ret2 == True:
            #corn_res = corners.reshape(-1, 2).astype('float32')
            right_imgpoints.append((right_corners.T[:,0,:]).astype('float32'))
            right_objpoints.append((right_objpoint.T).astype(int))
            #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #make left and right list to total list (imgpoints, objpoints)
        x_tot = np.concatenate((left_imgpoints[i][0], right_imgpoints[i][0])).tolist()
        y_tot = np.concatenate((left_imgpoints[i][1], right_imgpoints[i][1]).tolist()

        pre_tot_imgpoints = np.zeros((2, len(x_tot)))
        pre_tot_imgpoints[0], pre_tot_imgpoints[1]= x_tot, y_tot
        tot_imgpoints.append(pre_tot_imgpoints)

        pre_tot_objpoints = np.vstack([left_objpoints[i].T,right_objpoints[i].T]).T
        tot_objpoints.append(pre_tot_objpoints)

        #try imagepoints graph
        img_fit = np.concatenate(tot_imgpoints, axis=1)
        img_length = np.cumsum([x.shape[1] for x in tot_imgpoints])
        i_points = np.split(img_fit, img_length[:-1], axis=1)

        #print(tot_objpoints)

        #plt.plot(i_points[i][0, :][80:], i_points[i][1, :][80:], "o", color="#dae772", label="Image points")
        #plt.plot(tot_objpoints[i][2][80:], tot_objpoints[i][1][80:], "rx", label="Fitted World points")

       #plt.show()

    return images,  tot_imgpoints, tot_objpoints
