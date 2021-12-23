import cv2
import numpy as np
from numpy.core.defchararray import array
#import matplotlib.pyplot as plt
# import speech_recognition as sr
# import time
# from gtts import gTTS
# import os
from scipy.spatial import distance as dist
from collections import OrderedDict
from contextlib import nullcontext
import sys
import cv2 as cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape
import scipy
from skimage import data
from skimage.color import rgb2gray

import matplotlib.pyplot as pp

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import seaborn as sns
from scipy.signal import argrelextrema
from scipy.signal import argrelmin
from scipy.signal import argrelmax
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import skimage.io
from numpy import ones, vstack
from numpy.linalg import lstsq
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageEnhance
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

################################################### old start

def Slope(y2, y1, x2, x1):
    m = (y2 - y1)/(x2-x1)
    return m


def canny(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 0, 70)
    return canny


def GroupLines(image, lines8, labels):
    line_image = np.zeros_like(image)
    for l in labels:
        if(l == 0):
            x1, y1, x2, y2 = lines8[l].reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # red
        elif(l == 1):
            x1, y1, x2, y2 = lines8[l].reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue
        elif(l == 2):
            x1, y1, x2, y2 = lines8[l].reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # green
        elif(l == 3):
            x1, y1, x2, y2 = lines8[l].reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # pink

    return line_image


def draw_lines_on_image(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image


def threshold_color(image):
    lowerbound = (145, 145, 145)
    upperbound = (255, 255, 255)
    mask = cv2.inRange(image, lowerbound, upperbound)
    return mask


def getBackgroundImage(videoName):
    array = []
    cap = cv2.VideoCapture(videoName)
    frameCount = 60
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0  # used for iteration
    ret = True
    while (fc < frameCount and ret):
        ret, img = cap.read()
        cap.read()
        cap.read()
        cap.read()
        gray = rgb2gray(img)
        array.append(gray)
        fc += 1
    cap.release()
    array = np.asarray(array, dtype=np.float16)
    med = np.median(array, axis=0)
    med = np.asarray(med, dtype=np.float32)
    return med


region_of_interest_vertices = [
    #(992, 232),
    (829, 106),
    #(1322, 232),
    (1163, 106),
    #(1952, 1296),
    (1163, 1080),
    #(285, 1296)
    (268, 1080)
]


region_of_interest_vertices_for_lanes = [
    (842, 106),
    (1140, 106),
    (1669, 1080),
    (323, 1080)
]


def get_hi_n_lo_y(box):
    hi = 0
    lo = 0
    for point in box:
        _, y = point
        if(y > hi):
            lo = hi
            hi = y
    return hi, lo


def region_of_interest(img, vertices):

    mask = np.zeros_like(img)
    match_mask_color = 255

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hole_filling(img):
    floddedImg = img.copy()
    h, w = floddedImg.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(floddedImg, mask, (0, 0), 255)
    invOfFlodded = cv2.bitwise_not(floddedImg)
    return invOfFlodded


def get_m_c(line):
    x1, y1, x2, y2 = line.reshape(4)
    P = [x1, y1]
    Q = [x2, y2]
    points = [(x1, y1), (x2, y2)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords,rcond=-1)[0]
    return m,c

def get_average_line(lines,img):

    val=scipy.stats.mode(lines)
    
    h, w = img.shape
    x,_=val
    y1=0
    y2=h
    print(x[0])
    return x,y1,x,y2

def get_lines(img):
    FilteredLines = []
    UltimateLines = np.array(FilteredLines)
    lines = cv2.HoughLinesP(img, 6, np.pi/360, 100,
                            np.array([]), minLineLength=50, maxLineGap=1000)
    return lines


def get_inv(img):
    pts1 = np.float32([[842, 106], [1140, 106], [323, 1080],[1669, 1080]])
    pts2 = np.float32([[0, 0], [298, 0], [0, 947], [298, 947]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix2 = cv2.getPerspectiveTransform(pts2, pts1)

    inverse_prespective = cv2.warpPerspective(
        img, matrix, (298, 947))
    return img


def filterLines(lines):

    flag = 0
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            deltaY = abs(y2 - y1)
            deltaX = abs(x2 - x1)
            angleInDegrees = math.atan2(deltaY, deltaX) * 180 / math.pi
            if angleInDegrees > 85:
                if flag == 0:
                    flag = 1
                    filtered_lines = np.array([line])
                else:
                    filtered_lines = np.append(filtered_lines, [line])
    filtered_lines = filtered_lines.reshape(-1, 1, 4)
    return filtered_lines


def findCoutours(img, height=0, area=0):

    contours, _ = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImg = np.zeros_like(img)
    filtered_cnts = []
    bounding_pic = contoursImg.copy()
    flag = 0
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        a = cv2.contourArea(cnt)
        if h > height and a > area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            filtered_cnts = np.append(filtered_cnts, box)
            cv2.drawContours(bounding_pic, [box], 0, (255, 255, 255), 1)
            cv2.drawContours(contoursImg, [cnt], -1, (255, 255, 255), 1)
            #cv2.rectangle(bounding_pic, (x, y), (x+w, y+h), (255, 255, 255), 1)
    bounding_pic = bounding_pic | contoursImg
    filtered_cnts = filtered_cnts.reshape(-1, 4, 2)
    filtered_cnts = filtered_cnts.astype(int)
    return contoursImg, filtered_cnts, bounding_pic


def findCoutours_again(img, height=0, area=0):  # merge with uper wala

    contours, _ = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImg = np.zeros_like(img)
    filtered_cnts = []
    bounding_pic = contoursImg.copy()
    flag = 0
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        a = cv2.contourArea(cnt)
        if h > height and a > area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            filtered_cnts = np.append(filtered_cnts, x)
            filtered_cnts = np.append(filtered_cnts, y)
            filtered_cnts = np.append(filtered_cnts, x+w)
            filtered_cnts = np.append(filtered_cnts, y)
            filtered_cnts = np.append(filtered_cnts, x+w)
            filtered_cnts = np.append(filtered_cnts, y+h)
            filtered_cnts = np.append(filtered_cnts, x)
            filtered_cnts = np.append(filtered_cnts, y+h)
            cv2.drawContours(bounding_pic, [box], 0, (255, 255, 255), 1)
            cv2.drawContours(contoursImg, [cnt], -1, (255, 255, 255), 1)
            #cv2.rectangle(bounding_pic, (x, y), (x+w, y+h), (255, 255, 255), 1)
    bounding_pic = bounding_pic | contoursImg
    filtered_cnts = filtered_cnts.reshape(-1, 4, 2)
    filtered_cnts = filtered_cnts.astype(int)
    return contoursImg, filtered_cnts, bounding_pic


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            deltaY = abs(y2 - y1)
            deltaX = abs(x2 - x1)
            angleInDegrees = math.atan2(deltaY, deltaX) * 180 / math.pi
            if angleInDegrees > 50:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return line_image


def display_lines2(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image


def mergeAndFilterLines(lines1, lines2, lines3, lines4, img):
    lines5 = np.append(lines1, lines2, 0)
    lines6 = np.append(lines5, lines3, 0)
    lines7 = np.append(lines6, lines4, 0)

    lines8 = filterLines(lines7)
    All_lanes = display_lines2(img, lines8)
    plt.imsave('All_lanes.jpg', All_lanes, cmap='gray')

    return lines8


def FocalLength(measured_distance, real_length, length_in_rf_image):
    focal_length = (length_in_rf_image * measured_distance) / real_length
    return focal_length


def Distance_finder(Focal_Length, real_marker_length, marker_length_in_frame):
    distance = (real_marker_length * Focal_Length)/marker_length_in_frame
    return distance




def get_inverse_presp(img):
    
    pts1 = np.float32([[842, 106], [1140, 106], [323, 1080],[1669, 1080]])
    pts2 = np.float32([[0, 0], [298, 0], [0, 947], [298, 947]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix2 = cv2.getPerspectiveTransform(pts2, pts1)

    inverse_prespective = cv2.warpPerspective(
        img, matrix, (298, 947))

    plt.imsave('inverse prespective.jpg',
               inverse_prespective, cmap='gray')

    return matrix,inverse_prespective


def lane_detection(inverse_prespective):
    
    lines_via_output_plpl = get_lines(
    inverse_prespective)

    lines_via_output_plpl=filterLines(lines_via_output_plpl)

    igg=draw_lines_on_image(inverse_prespective,lines_via_output_plpl)

    plt.imsave('iggg.jpg', igg, cmap='gray')

    Means_two_point_o(lines_via_output_plpl,inverse_prespective)

    return


def distanceEstimation(inverse_prespective):


    _, boxes, countour_image = findCoutours_again(inverse_prespective)


    
    width = inverse_prespective.shape[1]
    res = np.zeros_like(countour_image)
    for box in boxes:
        cv2.drawContours(res, [box], 0, (255, 255, 255), 1)
        _, _, _, pp = box
        x, _ = pp
        if(x < width/2):
            y1, y2 = get_hi_n_lo_y(box)
    
    return 


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def get_cluster_id(point,points,labels):
    points_list=points.tolist()
    index=0
    for p in points:
        x1,y1=p
        x2,y2=point
        if(x1==x2 and y1==y2):
            break
        index=index+1
    return labels[index]

def Means_two_point_o(AllLines, img):
    all_points=[]
    for line in AllLines:
        sp,ep=line.reshape(-1,2)
        x1,y1=sp
        x2,y2=ep
        line_points=get_line(x1,y1,x2,y2)
        ln=[]
        count=0
        for p in line_points:
            x,y=p
            if x==x1 and y==y1:
                all_points=np.append(all_points,x)
                all_points=np.append(all_points,y)
            elif x==x2 and y==y2:
                all_points=np.append(all_points,x)
                all_points=np.append(all_points,y)
            elif (count %4)==0:
                all_points=np.append(all_points,x)
                all_points=np.append(all_points,y)
            count=count+1

    all_points=all_points.reshape(-1,2)

    all_points = all_points.astype(int)

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
    cluster.fit_predict(all_points)

    plt.scatter(all_points[:,0],all_points[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show()
    #clustering done

    c_0=[]
    c_1=[]
    for c_id,point in zip(cluster.labels_,all_points):
        x,y=point
        if c_id==0:
            c_0=np.append(c_0,x)
        if c_id==1:
            c_1=np.append(c_1,x)

    
    c_0 = c_0.astype(int)
    c_1 = c_1.astype(int)

    x11,y11,x12,y12=get_average_line(c_0,img)
    x21,y21,x22,y22=get_average_line(c_1,img)

    final_avg_lanes = np.zeros_like(img)

    
    cv2.line(final_avg_lanes,(int(x11),int(y11)),(int(x12),int(y12)),(255,255,255),1)
    cv2.line(final_avg_lanes,(int(x21),int(y21)),(int(x22),int(y22)),(255,255,255),1)
    

    plt.imsave('final_avg_lanes.jpg', final_avg_lanes, cmap='gray')

    plt.show()




def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
################################################### old end
coco_class_dic_with_index = {
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorbike': 4,
    'aeroplane': 5,
    'bus': 6,
    'train': 7,
    'truck': 8,
    'boat': 9,
    'bench': 14,
    'bird': 15,
    'cat': 16,
    'dog': 17,
    'horse': 18,
    'sheep': 19,
    'cow': 20,
    'elephant': 21,
    'zebra': 23,
    'giraffe': 24,
    'backpack': 25,
    'umbrella': 26,
    'handbag': 27,
    'tie': 28,
    'suitcase': 29,
    'frisbee': 30,
    'skis': 31,
    'snowboard': 32,
    'kite': 34,
    'skateboard': 37,
    'surfboard': 38,
    'bottle': 40,
    'cup': 42,
    'fork': 43,
    'knife': 44,
    'spoon': 45,
    'bowl': 46,
    'banana': 47,
    'apple': 48,
    'sandwich': 49,
    'orange': 50,
    'broccoli': 51,
    'carrot': 52,
    'pizza': 54,
    'doughnut': 55,
    'cake': 56,
    'chair': 57,
    'sofa': 58,
    'potted plant': 59,
    'bed': 60,
    'dining table': 61,
    'toilet': 62,
    'laptop': 64,
    'mouse': 65,
    'remote': 66,
    'keyboard': 67,
    'microwave': 69,
    'oven': 70,
    'toaster': 71,
    'sink': 72,
    'refrigerator': 73,
    'book': 74,
    'clock': 75,
    'vase': 76,
    'scissors': 77,
    'toothbrush': 80,
    'teddy bear': 78,
    'hair drier': 79,
    'tv monitor': 63,
    'cell phone': 68,
    'hot dog': 53,
    'wine glass': 41,
    'tennis racket': 39,
    'baseball bat': 35,
    'baseball glove': 36,
    'sports ball': 33,
    'traffic light': 10,
    'fire hydrant': 11,
    'stop sign': 12,
    'parking meter': 13
}
coco_class_dic_withspaces = {
    'teddy': -1, 'bear': 78,
    'hair': -1, 'drier': 79,
    'tv': -1, 'monitor': 63,
    'cell': -1, 'phone': 68,
    'hot': -1, 'dog': 53,
    'wine': -1, 'glass': 41,
    'tennis': -1, 'racket': 39,
    'baseball': -1, 'bat': 35, 'glove': 36,
    'sports': -1, 'ball': 33,
    'traffic': -1, 'light': 10,
    'fire': -1, 'hydrant': 11,
    'stop': -1, 'sign': 12,
    'parking': -1, 'meter': 13,
    'dining': -1, 'table': 61,
    'potted': -1, 'plant': 59
}

def get_center(x1,y1,x2,y2):
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        return cX, cY
    


def get_index(xc,yc,trackedobjects):
    index=0
    for (objectID, centroid) in trackedobjects.items():
        x = centroid[0]
        y = centroid[1]
        if x==xc and y==yc:
            return objectID
    return -1

class CentroidTracker():
	def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def register(self, centroid):
    		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def update(self, rects):
    	# check to see if the list of input bounding box rectangles
		# is empty
        
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
			return self.objects
        		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
            # centroids

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]
            			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)
			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# return the set of trackable objects
		return self.objects




# functons :
def get_all_coco_class_names(input):  # input is a list#this is the funtion needed to extract in main project
    # splitting by individual objects :
    # this will store all objects individualy :
    total_raw_input = input.split()  # total_raw_input is a list
    print('FROM get_all_coco_class_names :-> \n the given input was : ' + str(total_raw_input))
    # new object holder :
    objects = {'obj1': 0}
    # objects = ['obj1']
    objects.pop('obj1')  # poping a temp variable
    # now we need to check if these objects are in the coco class :
    for index, object in enumerate(total_raw_input):
        # check to see if the word is in the diconary with spaces :
        if coco_class_dic_withspaces.get(object, 0) < 0:
            # it is one of the words with spaces : checking the index right after the current one
            # need to cater for out of bounds in this segment :
            if coco_class_dic_withspaces.get(total_raw_input[index + 1], 0) > 0:
                # the next word is in the dic so we need to :
                # pop both words and concatinate as one with spaces :
                holder = str(object) + " " + str(total_raw_input[index + 1])

                # print(holder)
                # add it to the list of words :
                objects[holder] = coco_class_dic_withspaces.get(total_raw_input[index + 1], 0)
                # objects.append(holder)
        # checking the word in the regular dic :
        elif coco_class_dic_with_index.get(object, 0) > 0:
            # the object was found in class
            # if the objects class name is with spacing then pop the firs one and add it in the next index :
            # print(object)

            objects[object] = coco_class_dic_with_index.get(object, 0)
            # objects.append(object)  # saving the word found in the list in the finial list.
        # ele s  leave keep the loop going
    return objects
def TextToSpeech(results):#this is the funtion needed to extract in main project
    # text to speech
    myText = results
    language = 'en'
    Output = gTTS(text=myText, lang=language, slow=False)
    Output.save("audio.mp3")
    os.system("start audio.mp3")
    return results

def RecognizeSpeechAndReturnObjectsAndIndexs():#this is the funtion needed to extract in main project
    # making instances of the reconigser class:
    r2 = sr.Recognizer()  # to get the objects names
    r3 = sr.Recognizer()  # to get first input
    # telling where the input is conming from :
    with sr.Microphone() as source:
        print('say keyword "Find" then say the object to locate')
        time.sleep(1)  # wait 1 sec to let the user read and react.
        # adjusting the to the current ambient noise of the room :
        while True:
            r3.adjust_for_ambient_noise(source, duration=0.5)
            print('speak now')
            audio = r3.listen(source,timeout=5,phrase_time_limit=5)  # r3 will listen from the microphone and then store it in audio
            if 'find' in r3.recognize_google(
                    audio):  # r2 will comvert to text to speach and then the if condition will  see if the text is the keyword we choose
                # the keyword has been found and hence we will work accordingly :
                r2 = sr.Recognizer()
                print("keyword has been said")
                # getting an other input from the user after using the keyword :
                with sr.Microphone() as source:
                    print('keyword found say object : ')
                    audio_objects = r2.listen(source,timeout=5,phrase_time_limit=5)

                    try:
                        output = r2.recognize_google(audio_objects)
                        print(output)
                        # the function below will take the input and then return a list with all the words in it :
                        res = get_all_coco_class_names(output.__str__())  # output.str is a string
                        #TextToSpeech(output)
                       # print('these are the objects we can find for you : ' + output.__str__()(res.keys()) + ' indexs for the keys :' + output.__str__()(res.values()))
                        #findObjects(output,)
                        return res

                        # working to make object the class to look for
                    # excepts for errors
                    except sr.UnknownValueError:
                        print("unknown value error")
                    except sr.RequestError as e:
                        print('Request error \n' + 'failed'.format(e))
            else:
                str = "Your desired object could Not Found"
                TextToSpeech(str)
                # print('Not Found')
               # print("could not find")
def findObjects(outputs,img,objectToSearchIndex,prevSideHorizontal,prevSideVertical):#this is the funtion needed to extract in main project
    hT, wT, cT = img.shape
    bounding_box_img = [] #moeez code
    #print("hT:",hT," wT:",wT)
    bbox = []
    nmsThreshold = 0.3  # lower = more agressvie
    confThreshold = 0.5
    wT_of_eachside=(wT/3)
    hT_of_eachside=(hT/3)
    wTarr=[0]*3
    hTarr=[0]*3
    #width
    wTarr[0] = 0
    wTarr[1] = wTarr[0] + wT_of_eachside
    wTarr[2] = wTarr[1] + wT_of_eachside
    #print("wTarr",wTarr)
    # height
    hTarr[0] = 0
    hTarr[1] = hTarr[0] + hT_of_eachside
    hTarr[2] = hTarr[1] + hT_of_eachside
    #print("hTarr",hTarr)
    classIds = []
    confs = []
    for output in outputs:
       for det in output:
            scores = det[5:]
            classId = objectToSearchIndex#np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                # condition for checking if before entry so prevhorizontal and prevVertical will be same to unique numbers so it shouldn't repeat itself
                #left 0 inner 0 1 2
                #mid 1 inner 0 1 2
                #inner 0 1 2
                mx=x+int(w/2)
                my=y+int(h/2)
                if mx >= wTarr[0] and mx <= wTarr[1]:#left unique prevhorizontal 0
                    if my >= hTarr[0] and my <= hTarr[1] and (prevSideVertical == -1 or (not (prevSideHorizontal==0 and prevSideVertical == 0))): #left up 0
                        prevSideHorizontal = 0
                        prevSideVertical=0
                        str="Object is in screen right up"
                        TextToSpeech(str)
                    elif my >= hTarr[1] and my <= hTarr[2] and (prevSideVertical == -1 or (not (prevSideHorizontal==0 and prevSideVertical == 1))): #left mid
                        prevSideHorizontal = 0
                        prevSideVertical=1
                        str=" Object is in screen right mid"
                        TextToSpeech(str)
                    else:
                     if my >= hTarr[2] and my <= hT and (prevSideVertical == -1 or(not (prevSideHorizontal==0 and prevSideVertical == 2))): #left down
                        prevSideHorizontal = 0
                        prevSideVertical=2
                        str="Object is in screen right down"
                        TextToSpeech(str)
                elif mx >= wTarr[1] and mx <= wTarr[2]:#mid unique prevhorizontal 1
                    if my >= hTarr[0] and my <= hTarr[1] and (prevSideVertical == -1 or (not (prevSideHorizontal==1 and prevSideVertical == 0))): #mid up
                        prevSideHorizontal = 1
                        prevSideVertical=0
                        str = "Object is in screen mid up"
                        TextToSpeech(str)
                    elif my >= hTarr[1] and my <= hTarr[2] and (prevSideVertical == -1 or (not (prevSideHorizontal==1 and prevSideVertical == 1))): #mid mid
                        prevSideHorizontal = 1
                        prevSideVertical=1
                        str = "Object is in screen mid"
                        TextToSpeech(str)
                    else:
                     if my >= hTarr[2] and my <= hT and (prevSideVertical == -1 or (not (prevSideHorizontal==1 and prevSideVertical == 2))): #mid down
                         prevSideHorizontal = 1
                         prevSideVertical = 2
                         str = "Object is in screen mid down"
                         TextToSpeech(str)
                else:
                    if mx >= wTarr[2] and mx <= wT:#right
                        if my >= hTarr[0] and my <= hTarr[1] and (prevSideVertical == -1 or  (not (prevSideHorizontal==2 and prevSideVertical == 0))): #right up
                            prevSideHorizontal = 2
                            prevSideVertical = 0
                            str = "Object is in screen left up"
                            TextToSpeech(str)
                        elif my >= hTarr[1] and my <= hTarr[2] and (prevSideVertical == -1 or  (not (prevSideHorizontal==2 and prevSideVertical == 1))): #right mid
                            prevSideHorizontal = 2
                            prevSideVertical = 1
                            str = "Object is in screen left mid"
                            TextToSpeech(str)
                        else:
                          if my >= hTarr[2] and my <= hT and (prevSideVertical == -1 or (not (prevSideHorizontal==2 and prevSideVertical == 2))): #right down
                            prevSideHorizontal = 2
                            prevSideVertical = 2
                            str = "Object is in screen left down"
                            TextToSpeech(str)

    #removing overlaping boxes:
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h) , (255,0,255) , 2)
        cv2.rectangle(img,(x+int(w/2),y+int(h/2)), (x+int(w/2)+1,y+int(h/2)+1), (255,0,255) , 2)

        cv2.rectangle(img,(int(wTarr[0]),int(hTarr[0])),(int(wTarr[1]),img.shape[1]) , (255,0,0) , 2)
        cv2.rectangle(img,(int(wTarr[1]),int(hTarr[0])),(int(wTarr[2]),img.shape[1]) , (0,0,255) , 2)

        cv2.rectangle(img,(int(wTarr[0]),int(hTarr[0])),(wT,int(hTarr[1])) , (0,0,255) , 2)
        cv2.rectangle(img,(int(wTarr[0]),int(hTarr[1])),(wT,int(hTarr[2])) , (0,0,255) , 2)

        bounding_box_img = img[y:(y + h), x:(x + w)]  # sending the co ordinates to corp to #moeez code
        #printing name and confidance value:
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        # cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX, 0.6 ,(255,0, 255), 1)
    return bounding_box_img,prevSideHorizontal,prevSideVertical





# capture = cv2.VideoCapture(0)
classes_File_name='ClassName.txt'
classNames=[]
#yolo attributes:
width_height_Target =320
#read file
with open(classes_File_name,'rt') as f:
    classNames=f.read().rsplit('\n')

modelConfiguration= 'yolov3-custom.cfg'
modelWeights= 'yolov3-custom_final.weights'
#network ->
net= cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)#------------------------------isko coda kerke dekhna

# res = RecognizeSpeechAndReturnObjectsAndIndexs()  # speech recognition----returns dictionary of names and indexs
# prevSideHorizontal = -1  # for direction
# prevSideVertical = -1  # for direction

# for key in res:
#     objectToSearchIndex=res[key]-1
#     print("objIndex",objectToSearchIndex)


# while(capture.isOpened()):
# successfully_Retrieved, image_Captured = capture.read()
# cv2.imshow("capture",image_Captured)
#IMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP




backgrundImg = getBackgroundImage('V1.mkv')
backgrundImg = (backgrundImg*255).astype(np.uint8)  # comverting the
cv2.imshow('background', backgrundImg)
plt.imsave('background.jpg', backgrundImg, cmap='gray')


backgrundImg = cv2.imread("background.jpg", 0)


canny = canny(backgrundImg)
plt.imsave('canny.jpg', canny, cmap='gray')


cropped_image = region_of_interest(canny, np.array(
    [region_of_interest_vertices], np.int32),)



# erode dialate image
kernal = np.ones((3, 3), np.uint8)
er = cv2.dilate(cropped_image, kernal, iterations=1)
cropped_image = cv2.erode(er, kernal, iterations=1)
plt.imsave('cropped.jpg', cropped_image, cmap='gray')

# hole filled image
filledImg = hole_filling(cropped_image)
#cv2.imshow('filled', filledImg)
plt.imsave('filled.jpg', filledImg, cmap='gray')

# hole + erode dialiate
filled_cropped_image = filledImg | cropped_image
plt.imsave('filled_cropped_image.jpg', filled_cropped_image, cmap='gray')





matrix,inverse_prespective_img=get_inverse_presp(output_plpl)



lane_detection(inverse_prespective_img)
distanceEstimation(inverse_prespective_img)









def run_detection(videoName):
    cap = cv2.VideoCapture(videoName)
    tracker=CentroidTracker()
    fc = 0  # used for iteration
    
    ret = True
    while (ret):
        ret, image_Captured = cap.read()
        blob =cv2.dnn.blobFromImage(image_Captured,1/255,(width_height_Target,width_height_Target),[0,0,0],1,crop=False)#search blob function
        net.setInput(blob) #yolo ko image
        #names of the layer
        layerNames= net.getLayerNames()#layer names
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]  #names of output of the layer
        # print(outputNames) #names of output of the layer
        outputs=net.forward(outputNames)#search this too----------------------------
        #moeez code start
        # bounding_box_img,prevSideHorizontal,prevSideVertical = findObjects(outputs, image_Captured,objectToSearchIndex,prevSideHorizontal,prevSideVertical)
        hT, wT, cT = image_Captured.shape
        bounding_box_img = [] #moeez code
        bbox = []
        nmsThreshold = 0.3  # lower = more agressvie
        confThreshold = 0.5
        classIds = []
        confs = []
        #IMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        for output in outputs:
            for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        w,h = int(det[2]*wT),int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                        bbox.append([x, y, w, h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
        boxes=[]
        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            # x=+6
            # y=+6
            # w=-6
            # h=-6
            boxes.append(x)
            boxes.append(y)
            boxes.append(w+x)
            boxes.append(h+y)
            cv2.putText(image_Captured, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            cv2.rectangle(image_Captured,(x,y),(x+w,y+h) , (255,0,255) , 2)
        boxes=np.array(boxes)
        boxes=boxes.reshape(-1,4)
        boxes=list(boxes)

        trackedobjects=tracker.update(boxes)



        for (objectID, centroid) in trackedobjects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(image_Captured, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image_Captured, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        TrackedCars = []
        for box in boxes:
            x1,y1,x2,y2  = box
            xc,yc = get_center(x1,y1,x2,y2)
            index = get_index(xc,yc,trackedobjects)
            TrackedCars = np.append(TrackedCars,(x1,y1,x2,y2))


        

        cv2.imshow("Image", image_Captured)
        cv2.waitKey(1)

    cap.release()



print('on it')

run_detection("V1.mkv")








# try:
#     cv2.imshow("cropped", bounding_box_img)
# except:
#     print("not viewed")
#     #moeez code end
# #for now only taike one object for just test case
# #it should complete the process and then go to another object
# # findObjects(outputs,image_Captured,objectToSearchIndex)

# # cv2.imshow("Image",image_Captured)
# cv2.waitKey(1)
# if cv2.waitKey(1) & 0xFF == ord('q'): #set to quit on pressing q
#     break

# capture.release() # After the loop release the cap object
# cv2.destroyAllWindows() # Destroy all the windows



