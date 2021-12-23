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


# Routine to fix


# is a commented line, prolly not workin or of no use
#
#  is good line commented to speed up debugging
#
#
# chunk of lines immediately after this are for speeding up work









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
    frameCount = 580
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
    (548, 130),
    #(1322, 232),
    (727, 130),
    #(1952, 1296),
    (1050, 720),
    #(285, 1296)
    (200, 720)
]


region_of_interest_vertices_for_lanes = [
    (558, 130),
    (717, 130),
    (1024, 720),
    (224, 720)
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
    pts1 = np.float32([[992, 232], [1306, 232], [360, 1296],[1885, 1296]])
    pts2 = np.float32([[0, 0], [314, 0], [0, 1064], [314, 1064]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix2 = cv2.getPerspectiveTransform(pts2, pts1)

    inverse_prespective = cv2.warpPerspective(
        img, matrix, (314, 1064))
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
    
    pts1 = np.float32([[558,130 ], [717,130 ], [224, 720], [1024, 720]])
    pts2 = np.float32([[0, 0], [159, 0], [0, 590], [159, 590]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    inverse_prespective = cv2.warpPerspective(
        img, matrix, (159, 590))

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
    
    return matrix


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

# start from here

backgrundImg = getBackgroundImage('footage.mp4')
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




cropped_filled_cropped_image = region_of_interest(filled_cropped_image, np.array(
    [region_of_interest_vertices_for_lanes], np.int32),)

marker_contours_filled_cropped_image, lanemarkers, Lane_boundingImg = findCoutours(
    cropped_filled_cropped_image, 8, 20)
plt.imsave('marker_contours_filled_cropped_image.jpg',
           marker_contours_filled_cropped_image, cmap='gray')
plt.imsave('Lane_boundingImg.jpg',
           Lane_boundingImg, cmap='gray')


contoured_lane_markers = np.zeros_like(backgrundImg)

for marker in lanemarkers:
    cv2.drawContours(contoured_lane_markers, [marker], 0, (255, 255, 255), 1)

plt.imsave('contoured_lane_markers.jpg',
           contoured_lane_markers, cmap='gray')

output_plpl = hole_filling(contoured_lane_markers)


plt.imsave('output_plpl.jpg',
            output_plpl, cmap='gray')


_,koko=get_inverse_presp(backgrundImg)

plt.imsave('koko.jpg',
            koko, cmap='gray')


matrix,inverse_prespective_img=get_inverse_presp(output_plpl)



lane_detection(inverse_prespective_img)
distanceEstimation(inverse_prespective_img)
