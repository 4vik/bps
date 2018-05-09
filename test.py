import cv2
import numpy as np


def detect_circle(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = img.shape

    square_img = cv2.resize(img, (height, height))
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                               50,
                               minDist=1000000,
                               param1=100,
                               param2=400,
                               minRadius=int(height * 0.25),
                               maxRadius=int(height * 0.5))

    if circles is None:
        cv2.imshow('no circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 3)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', square_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


detect_circle('circle.png')
# detect_circle('eye.jpg')
# detect_circle('eyes2.png')
