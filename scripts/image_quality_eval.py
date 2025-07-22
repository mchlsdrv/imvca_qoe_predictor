import PIL.Image
from imquality import brisque
import pathlib
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
img_pth = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/la_kost3.jpg')

# img = cv2.imread(str(img_pth))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = PIL.Image.fromarray(img)
img = Image.open(img_pth)
# plt.imshow(img)
brisque.score(img)
