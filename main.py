import numpy as np
import cv2
import time

def get_size(img):
	return list(img.shape)[:2]

def timeit(func):
	def wrapper(*args, **kw):
		time1 = time.time()
		result = func(*args, **kw)
		time2 = time.time()
		print(func.__name__, time2-time1)
		return result
	return wrapper

class ShadowRemoval_Client:
	def __init__(self, img):
		self.img = np.asarray(img, np.float32) # The image to be handled;
		self.img2 = img # The real image;
		self.rows, self.cols = get_size(img)
		self.mask_s = np.zeros((self.rows, self.cols), dtype = np.uint) # The area that is entirely inside the shadow;
		self.mask_l = np.zeros((self.rows, self.cols), dtype = np.uint) # The area that is entirely outside the shsdow;
		self.mask_shadow = np.zeros((self.rows, self.cols), dtype = np.uint) # The area where shadow removal is required;

		self._SHADOW = 1 # The flag of shadow;
		self._threshold = 0.1;
		self._drawing = True # The flag of drawing;
		self._drawn = False # The status of whether seed initialise is finished;


	def init_mask(self, event, x, y, flags, param):
		self._thickness = 3 # The thickness in drawing;
		self._WHITE = [255, 255, 255] # Pure white;

		# Draw a point on the image;
		if event == cv2.EVENT_RBUTTONDOWN:
			if self._drawing == True:
				cv2.circle(self.img, (x, y), self._thickness, self._WHITE, -1)
				self.mask_s[y-self._thickness:y-self._thickness, x-self._thickness:x+self._thickness] = self._SHADOW
				self._shadow_seed = self.img[y-self._thickness:y-self._thickness, x-self._thickness:x+self._thickness].copy()

		elif event == cv2.EVENT_RBUTTONUP:
			if self._drawing == True:
				self._drawing = False
				self._drawn = True
				cv2.circle(self.img, (x, y), self._thickness, self._WHITE, -1)

	def _calc_invariant_distance(self, pixel1, pixel2):
		'''Using the formula in the paper;
			Invariant distance between two RGB colors is 1-cos(theta), where theta is the angle between there corresponding 3-vectors.'''
		'''Using Cosine Formula;'''
		'''Intuition:具有相似反射性质的像素点均位于一条直线附近;'''
		a = (pixel1*pixel1).sum()
		b = (pixel2*pixel2).sum()
		dis = pixel1 - pixel2
		c = (dis*dis).sum()
		return 1-abs((a+b-c)/(2*np.sqrt(a)*np.sqrt(b)))

	def _get_dist(self):
		if self._drawn == True:
			vf = np.vectorize(self._calc_invariant_distance)
			self._dist = vf(self.img, self._shadow_seed.sum(axis = 0).sum(axis = 0)/(self._shadow_seed.size / 3))

	def _region_growing(self):
		pass

	def _detection(self):
		pass

	def run(self):
		pass

if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\ad.jpg')
	SR = ShadowRemoval_Client(img)
	cv2.namedWindow('input')
	a = cv2.setMouseCallback('input',SR.init_mask)

	count = 0

	while(1):
		cv2.imshow('input', np.asarray(SR.img, dtype = np.uint8))
		k = 0xFF & cv2.waitKey(1)

		if k == 27:
			break
		elif k == ord('n'):
			SR._get_dist()
			print(count)
		
	print(SR._dist)
	cv2.destroyAllWindows()


	