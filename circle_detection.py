import cv2
import numpy as np


def detect_circle(image):
    captcha_path = "captchas/"
    img = cv2.imread(captcha_path + image, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    square_img = cv2.resize(img, (height, height))

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(square_img, cv2.HOUGH_GRADIENT,
                               2,
                               minDist=1000000,
                               param1=10,
                               param2=1,
                               minRadius=int(height * 0.25 * 0.5),
                               maxRadius=int(height * 0.5 * 0.5))

    if circles is None:
        cv2.imshow('no circles', square_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(square_img, (i[0], i[1]), i[2], (0, 255, 0), 3)
            # draw the center of the circle
            cv2.circle(square_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles in ' + image, square_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


detect_circle('captcha_1.jpg')
detect_circle('captcha_2.jpg')
detect_circle('captcha_3.jpg')
detect_circle('captcha_4.jpg')
