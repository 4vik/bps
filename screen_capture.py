import numpy as np
import cv2
from PIL import ImageGrab


def capture_image_on_change():
    img_np_prev = None
    img_np = None

    change_prev = 0
    change = 0

    while True:
        img_np_prev = img_np
        change_prev = change
        # capture computer screen
        img = ImageGrab.grab(bbox=(10, 10, 400, 400))
        # convert image to numpy array
        img_np = np.array(img)

        if img_np_prev is not None:
            # Compute 1-Norm between img and previous image
            change = np.sum(np.abs(img_np_prev - img_np))
        if change != change_prev:
            print(change)
        # convert color space from BGR to RGB
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # show image on OpenCV frame
        cv2.imshow("Screen", frame)
        # write frame to video writer
        # out.write(frame)
        if cv2.waitKey(1) == 27:
            break

    # out.release()
    cv2.destroyAllWindows()

capture_image_on_change()