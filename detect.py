import numpy as np
import cv2
import time
from main import timeit, get_size

def detect(img, seed):
	shape = tuple(get_size(img))
	[Ms, Ml, Mshadow] = [np.zeros(shape) for i in range(3)]


def test(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		print(x, y)

if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\test.jpg')
	cv2.namedWindow('input')
	cv2.setMouseCallback('input', test)
	cv2.imshow('input', img)
	while(1):
		k = 0xFF & cv2.waitKey(1)
		if k == 27:
			cv2.destroyAllWindows()
			break