import numpy as np
import cv2 as cv
import pandas as pd

img = cv.imread('v2.jpg', cv.IMREAD_GRAYSCALE)
boolmat = img > 250
#boolmat=boolmat.T
frame=pd.DataFrame(boolmat.astype(int))
frame.to_csv('wall.csv',header=None,index=None)