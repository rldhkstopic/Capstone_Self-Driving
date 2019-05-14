import matplotlib.image as mpimg
import numpy as np
import os
import io
import cv2
import math
import glob
import pickle

from moviepy.editor import VideoFileClip
from scipy.misc import imresize
from imgaug import augmenters as iaa

def grayscale(img):
    """Applies the Grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
        """Applies an image mask."""
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255, ) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        # vertiecs로 만든 polygon으로 이미지의 ROI를 정하고 ROI 이외의 영역은 모두 검정색으로 정한다.

        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    [workflow]
    1) Hough 변환으로 생성된 모든 라인들을 조사하고 각 라인들의 기울기를 통해 왼쪽과 오른쪽 중 어디에 위치해있는 지를 결정한다.
    -> Left Lane : Negative Slope , Right Lane : Positive Slope
    2) track extrema : 트랙의 극점을 찾는다.
    3) compute averages : 모든 평균값을 합친다.
    4) solve for b intercept (y=mx+b)
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame

    y_global_min = img.shape[0] # min은 y값 중 가장 큰값을 의미할 것이다. 또는 차로부터 멀어져 길을 따라 내려가는 지점 (?)
    y_max = img.shape[0]

    l_slope, r_slope = [], []
    l_lane,r_lane = [], []
    det_slope = 0.5
    α = 0.2

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)

        y_global_min = min(y1,y2,y_global_min)

    if((len(l_lane) == 0) or (len(r_lane) == 0)): # 오류 방지
        print ('no lane detected')
        return 1
    else:
        print('Successfully Lane Detected')

    # Slope
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)

    if l_slope_mean or r_slope_mean is NaN:
        pass

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1

    # y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    l_x1 = ((y_global_min - l_b)/l_slope_mean)
    l_x2 = ((y_max - l_b)/l_slope_mean)

    r_x1 = ((y_global_min - r_b)/r_slope_mean)
    r_x2 = ((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1: # Left line이 Right Line보다 오른쪽에 있는 경우 (Error)
        l_x1 = ((l_x1 + r_x1)/2)
        r_x1 = l_x1

        l_y1 = ((l_slope_mean * l_x1 ) + l_b)
        r_y1 = ((r_slope_mean * r_x1 ) + r_b)

        l_y2 = ((l_slope_mean * l_x2 ) + l_b)
        r_y2 = ((r_slope_mean * r_x2 ) + r_b)

    else: # l_x1 < r_x1 (Normal)
        l_y1 = y_global_min
        l_y2 = y_max

        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0

    else:
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*current_frame

    margin = 10

    cv2.line(img, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), color, thickness)
    cv2.line(img, (next_frame[4], next_frame[5]), (next_frame[6], next_frame[7]), color, thickness)
    cv2.line(img, (next_frame[0], next_frame[1]), (next_frame[4], next_frame[5]), color, thickness)
    # cv2.polylines(img, [pts], True, (0, 255, 255), 2)

    cache = next_frame

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    image = imresize(image, (720, 1280, 3))

    global first_frame

    kernel_size = 3

    # Canny Edget Detection Threshold
    low_thresh = 100
    high_thresh = 150

    rho = 4
    theta = np.pi/180
    thresh = 100
    min_line_len = 150
    max_line_gap = 180

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    # lower_white = np.array([0, 0, 170], dtype = "uint8")
    # upper_white = np.array([255, 255, 255], dtype = "uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow) # 흰색과 노란색의 영역을 합친다.
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw) # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(image)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return result, roi_image

def get_pts(image):
    height, width = image.shape[:2]

    vertices = np.array([
                [( 0/12) * width, (7.0/7) * height],
                [( 0/12) * width, (6.0/7) * height],
                [( 5/12) * width, (4.0/7) * height], #기울기 1/2
                [( 7/12) * width, (4.0/7) * height],
                [(12/12) * width, (6.0/7) * height],
                [(12/12) * width, (7.0/7) * height],], dtype = np.int32)
    return vertices

def get_roi(image):
    vertices = [get_pts(image)]
    roi_image = region_of_interest(canny_edges, vertices)
    return roi_image

def add_lines(image):
    line_image = np.zeros_like(image)
    pts = get_pts(image)
    pts = pts.reshape((-1, 1, 2))

    lines = cv2.polylines(line_image, [pts], True, (255, 0, 0), 3)
    result = cv2.addWeighted(image, 1, lines, 1, 0)
    return result

"""--------------------------image test--------------------------------------"""
for source_img in os.listdir("test_images/"):
    first_frame = 1
    image = mpimg.imread("test_images/" + source_img)

    processed, roi_image = process_image(image)
    line_image = add_lines(image)

    mpimg.imsave("out_images/annotated/annotated_" + source_img, processed)
    mpimg.imsave("out_images/roi/roi_" + source_img, roi_image)
    mpimg.imsave("out_images/lines/lines_" + source_img, line_image)

"""--------------------------------------------------------------------------"""

"""--------------------------Video test--------------------------------------"""

def save_video(filename):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, 60.0, (1280,720))
    return out

first_frame = 1
cap = cv2.VideoCapture("test_videos/01.mp4")
# clip1 = save_video('out_videos/01_result.mp4') # result 영상 저장
# clip2 = save_video('out_videos/01_roi.mp4') # roi 영상 저장

while (cap.isOpened()):
    _, frame = cap.read()
    result, roi = process_image(frame)

    cv2.imshow("result", result)
    # clip1.write(result)
    cv2.imshow("roi", roi)
    # clip2.write(roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""--------------------------------------------------------------------------"""
