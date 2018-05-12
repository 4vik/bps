import numpy as np
import cv2
from PIL import ImageGrab
from PIL import Image
import pytesseract
import time
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

end=0

def capture_image_on_change():
    bbox_size=(80,300,700,700)
    img_np_prev = ImageGrab.grab(bbox=bbox_size)
    img_np = ImageGrab.grab(bbox=bbox_size)

    change = 0
	
    lower_black = np.array([0,0,0])
    upper_black = np.array([40,40,40])
	
    longest_vertical_line=np.zeros(4)
    longest_horizontal_line=np.zeros(4)
    coord=np.zeros(4)
    longest_vert=0
    longest_horiz=0
	
    while True:
        img_np_prev = img_np

        img = ImageGrab.grab(bbox=bbox_size)
        img_np = np.array(img)

        # Compute 1-Norm between img and previous image
        change = np.sum(np.abs(img_np_prev - img_np))
        if change >= 10000:
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            frame=cv2.inRange(frame,lower_black,upper_black)
            #frame = cv2.Canny(frame,50,150,apertureSize = 3)
            minLineLength=80
            lines = cv2.HoughLinesP(image=frame,rho=1,theta=np.pi/180, threshold=60,lines=np.array([]), minLineLength=minLineLength,maxLineGap=10)
            list_x=[]
            list_y=[]			
            if lines is not None:
                a,b,c = lines.shape
                for i in range(a):
                    coord[0]=lines[i][0][0]
                    coord[1]=lines[i][0][1]
                    coord[2]=lines[i][0][2]
                    coord[3]=lines[i][0][3]
                    list_x.append(coord[0])
                    list_x.append(coord[2])
                    list_y.append(coord[1])
                    list_y.append(coord[3])					
			    #DO CHUNNT DR RIESE TEIL WONI GRAD USGSCHNITTE HA UNTER "YAYAYAYA"
                #crop_img=img_np[int(min_y):int(max_y),int(min_x):int(max_x)]
                min_x=min(list_x)
                min_y=min(list_y)
                max_x=max(list_x)
                max_y=max(list_y)
                #cv2.circle(frame,(int(min_x),int(min_y)),10,(255,0,0))				
                #cv2.circle(frame,(int(max_x),int(max_y)),30,(255,0,0))				
                crop_img=img_np[int(min_y):int(max_y),int(min_x):int(max_x)]
                load_img_and_detect_circle(crop_img)
                break
                				
                
		# write frame to video writer
        # out.write(frame)
        if cv2.waitKey(1) == 27:
            break
			
    # out.release()
    cv2.destroyAllWindows()
	
def get_line_length(coord):
    return (coord[3]-coord[1])*(coord[3]-coord[1])+(coord[2]-coord[0])*(coord[2]-coord[0])


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


    # End of debug region
    #----------------------------------------------------------------------------

    if circles is not None:
      quadrant = compute_quadrant(img_grey_square, circles[0,0])
      sub_img = get_sub_image(image, quadrant)
      cv2.imshow('subimage ' + str(quadrant), sub_img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      lower_black = np.array([0,0,0])
      upper_black = np.array([40,40,40])
      sub_img=cv2.inRange(sub_img,lower_black,upper_black)
      img_for_ocr = Image.fromarray(np.array(sub_img))
      COIN_CODE = pytesseract.image_to_string(img_for_ocr)
      print(COIN_CODE)
	  
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
  img = img_name
  detect_circle(img)
  
capture_image_on_change()

