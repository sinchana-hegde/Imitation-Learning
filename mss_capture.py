import time
import numpy as np
import cv2
import mss
from PIL import Image
def abc():
  with mss.mss() as sct:
    monitor = {'top': 146, 'left': 1911, 'width': 400, 'height': 220}

    last_time = time.time()

    img = np.array(sct.grab(monitor))
    #print("in_mss")

    return img

def test():
with mss.mss() as sct:

    monitor = {'top': 146, 'left': 1910, 'width': 400, 'height': 220}
    last_time = time.time()
    img = np.array(sct.grab(monitor))
    img = cv2.resize(img,(100,100))
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		#theta = np.linspace(0., 180., max(img.shape), endpoint=False)
		#sinogram = radon(img, theta=theta, circle=False)
		#reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
		#img = np.array(reconstruction_fbp,dtype=np.int8)

    return img
