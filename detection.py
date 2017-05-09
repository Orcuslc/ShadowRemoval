import numpy as np
import cv2
import time
import sys  
sys.setrecursionlimit(1000000)
from main import timeit, get_size

### Configurations ###
DOWNSAMPLE = 2
SEED_ITER = 4
RATE = 0.5
CANNY = (150, 300)
SEED_TOL = 200
search_directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
MS_TOL = 300
######################

downsample = lambda img, rate = RATE: cv2.pyrDown(img, np.array(img[::int(1/rate),::int(1/rate),:], dtype=np.uint8), (int(img.shape[1]*rate), int(img.shape[0]*rate)))
upsample = lambda img, rate = int(1/RATE): cv2.pyrUp(img, np.zeros((img.shape[1]*2, img.shape[0]*2, 3), dtype = np.uint8), (img.shape[1]*2, img.shape[0]*2))
gray = lambda img: cv2.cvtColsbor(img, cv2.COLOR_BGR2GRAY)
dist = lambda x, y: sum((x-y)**2)
invariant_dist = lambda x, y: 1-abs(sum(x*y)/np.sqrt(sum(x*x)*sum(y*y)))

def grow_seed(img, seed_loc):
	tmp = np.array(img)
	for i in range(DOWNSAMPLE): # Downsample the original image
		img_ds = downsample(tmp)
		tmp = img_ds
	seed_loc_ds = (int(seed_loc[0]*RATE**DOWNSAMPLE), int(seed_loc[1]*RATE**DOWNSAMPLE))
	seed_pixel_ds = img_ds[seed_loc_ds]
	edges_ds = cv2.Canny(cv2.GaussianBlur(img_ds, (3, 3), 0), CANNY[0], CANNY[1])
	seed_mask_ds = np.zeros(get_size(img_ds))
	seed_mask_ds[seed_loc_ds] = 1
	visited = np.zeros(get_size(img_ds))
	def search(point, seed_pixel):
		if point[0] < 0 or point[1] < 0 or point[0] >= img_ds.shape[0] or point[1] >= img_ds.shape[1]:
			return
		if visited[point]:
			return
		elif edges_ds[point]:
			return
		elif(dist(img_ds[point], seed_pixel) < SEED_TOL):
			visited[point] = 1
			seed_mask_ds[point] = 1
			for i in range(4):
				search((point[0]+search_directions[i][0], point[1]+search_directions[i][1]), seed_pixel)
	for i in range(SEED_ITER):
		search(seed_loc_ds, seed_pixel_ds)
		seed_pixel_ds = np.mean(img_ds[np.where(seed_mask_ds == 1)], axis = 0)
		print(seed_pixel_ds)
		visited[:, :] = 0
	for i in range(DOWNSAMPLE):
		seed_mask_ds = upsample(seed_mask_ds)
	seed_mask_ds[np.where(seed_mask_ds > 0.5)] = 1
	seed_mask_ds[np.where(seed_mask_ds <= 0.5)] = 0
	return seed_mask_ds

def compute_mask(img, seed_mask, seed_loc):
	size = get_size(img)
	seed = np.where(seed_mask == 1)
	seed_color = np.median(img[seed], axis = 0)
	inv_dist = np.apply_along_axis(invariant_dist, 2, img+1e-6, seed_color)
	surface = np.zeros(size)
	surface[np.where(inv_dist < 0.01)] = 1
	Ms, Ml = np.zeros(size), np.ones(size)
	Ms[seed] = 1
	Ml[seed] = 0
	visited = np.zeros(size)
	global sd_ms, sd_ml
	sd_ms = 0
	sd_ml = 1000000
	def search(point):
		global sd_ms, sd_ml
		if sd_ms >= sd_ml:
			return
		if point[0] < 0 or point[1] < 0 or point[0] >= size[0] or point[1] >= size[1]:
			return
		if Ms[point] == 1:
			for i in range(4):
				search((point[0]+search_directions[i][0], point[1]+search_directions[i][1]))
		elif dist(seed_color, img[point]) < MS_TOL and surface[point] == 1:
			Ms[point] = 1
			Ml[point] = 0
			sd_ms = sum(np.std(img[np.where(Ms == 1)], axis = 0))
			sd_ml = sum(np.std(img[np.where(Ml == 1)], axis = 0))
			for i in range(4):
				search((point[0]+search_directions[i][0], point[1]+search_directions[i][1]))
	search(seed_loc)
	return surface

def dbclick(event, x, y, flags, param):
	global seed_loc
	if event == cv2.EVENT_LBUTTONDBLCLK:
		seed_loc = (y, x)

if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\test3.jpg')
	cv2.namedWindow('input')
	cv2.setMouseCallback('input', dbclick)
	cv2.imshow('input', img)
	cv2.moveWindow('input',img.shape[1]+10,90)
	cv2.namedWindow('seed')
	while(1):
		cv2.imshow('input',img)
		k = 0xFF & cv2.waitKey(1)
		if k == 27:	# ESC
			break
		elif k == ord('1'):
			seed_mask = grow_seed(img, seed_loc)
			res = compute_mask(img, seed_mask, seed_loc)
			cv2.imshow('seed', res)
		else:
			continue
