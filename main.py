import numpy as np
import cv2
import time

def get_size(img):
	return list(img.size)[:2]

def timeit(func)
	def wrapper(*args, **kw):
		time1 = time.time()
		result = func(*args, **kw)
		time2 = time.time()
		print(func.__name__, time2-time1)
		return result
	return wrapper

class ShadowRemoval_Client:
	def __init__(self, img):
		self.img = img
		self.rows, self.cols = get_size(img)
		self.mask_s = np.zeros((self.rows, self.cols), dtype = np.uint)
		self.mask_l = np.zeros((self.rows, self.cols), dtype = np.uint)
		self.mask_shadow = np.zeros((self.rows, self.cols), dtype = np.uint)

		self._SHADOW = 1

	def init_seed(self, event, x, y, flags, param):
		self._drawing = False
		self._thickness = 3
		self._WHITE = [255, 255, 255]

		# Draw a point on the image;
		if event == cv2.EVENT_RBUTTONDOWN:
			if self._drawing == True:
				cv2.circle(self.img, (x, y), self._thickness, self._WHITE, -1)

		elif event == cv2.EVENT_MOUSEMOVE:
			if self._drawing == True:
				cv2.circle(self.img, (x, y), self._thickness, self._WHITE, -1)

		elif event == cv2.EVENT_LBUTTONUP:
			if self._drawing == True:
				self._drawing = False
				cv2.circle(self.img, (x, y), self._thickness, self._WHITE, -1)

		self.mask_s[y, x] = self._SHADOW

	def _detection(self):
		
