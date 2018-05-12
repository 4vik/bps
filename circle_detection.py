import cv2
import numpy as np


def detect_circle(image):
    """ Returns the subimage of 'image' that is marked by the
        circle.
    """
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = img_grey.shape
    img_grey_square = cv2.resize(img_grey, (height, height))
    circles = cv2.HoughCircles(img_grey_square,
                              cv2.HOUGH_GRADIENT,
                              2,
                              minDist=1000000,
                              param1=10,
                              param2=1, 
                              minRadius=int(height * 0.25 * 0.5),
                              maxRadius=int(height * 0.5 * 0.5))
    #----------------------------------------------------------------------------
    # Begin of debug region
    if __debug__:
      img_col_square = cv2.resize(image, (height, height))
      if circles is None:
          cv2.imshow('no circles', img_col_square)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
          return
      else:
          for i in circles[0, :]:
              # draw the outer circle
              cv2.circle(img_col_square, (i[0], i[1]), i[2], (0, 255, 0), 3)
              # draw the center of the circle
              cv2.circle(img_col_square, (i[0], i[1]), 2, (0, 0, 255), 3)

      cv2.imshow('detected circles', img_col_square)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    # End of debug region
    #----------------------------------------------------------------------------

    if circles is not None:
      quadrant = compute_quadrant(img_grey_square, circles[0,0])
      sub_img = get_sub_image(image, quadrant)

    #----------------------------------------------------------------------------
    # Begin of debug region
    if __debug__:
      cv2.imshow('subimage ' + str(quadrant), sub_img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    # End of debug region
    #----------------------------------------------------------------------------


def compute_quadrant(image, circle):
  """ Q1 = Top-Right,  
      Q2 = Top-Left,
      Q3 = Bottom-Left,
      Q4 = Bottom-Right
  """
  height, width = image.shape
  cx, cy, _ = circle
  if cx < (width / 2):
    if cy < (height / 2):
      return 2
    else:
      return 3
  else:
    if cy < (height / 2):
      return 1
    else:
      return 4


def get_sub_image(image, quadrant):
  """ Extracts the respective quadrant of the image, w.r.t to
      image.width/2, image.height/2
  """
  height, width, _ = image.shape
  if quadrant == 1:
    return image[0: int(height / 2), int(width / 2):width]
  if quadrant == 2:
    return image[0: int(height / 2), 0:int(width / 2)]
  if quadrant == 3:
    return image[int(height / 2):height, 0:int(width / 2)]
  if quadrant == 4:
    return image[int(height / 2):height, int(width / 2):width]



def load_img_and_detect_circle(img_name):
  """ For debug purposes only. Loads the image from disc and 
      performs the circle detection.
  """
  captcha_path = "captchas/"
  img = img_name
  detect_circle(img)



#----------------------------------------------------------------------------
# Begin of debug region
if __debug__:
  load_img_and_detect_circle('captcha_1.jpg')
  load_img_and_detect_circle('captcha_2.jpg')
  load_img_and_detect_circle('captcha_3.jpg')
  load_img_and_detect_circle('captcha_4.jpg')
  print("Finished circle detection")
# End of debug region
#----------------------------------------------------------------------------