import cv2
import numpy as np

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                #cv2.circle(img,(x,y),3,(255,0,0),-1)
                self.points.append((x,y))

def click_origin(img):
    """find origin of dot board by clicking"""

    coordinateStore1 = CoordinateStore()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', coordinateStore1.select_point)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27: break

    cv2.destroyAllWindows()
    print(coordinateStore1.points)
    return coordinateStore1.points


"""
#instantiate class
coordinateStore1 = CoordinateStore()


# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',coordinateStore1.select_point)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


print ("Selected Coordinates: ")
#for i in coordinateStore1.points:
 #   print (i)

print(coordinateStore1.points)

"""
