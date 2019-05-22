from __future__ import division

from torch.autograd import Variable
from torch.cuda import FloatTensor
import torch.nn as nn

from darknet import Darknet, set_requires_grad
from shapely.geometry import Polygon, Point
from moviepy.editor import VideoFileClip
from scipy.misc import imresize
from PIL import Image

import numpy as np
import os
import io
import cv2
import pafy
import pickle
import argparse
import time
import serial
from math import *
from util import *

# Color
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)
dark = (1, 1, 1)

cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# Global 함수 초기화
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)


def arg_parse():
    parses = argparse.ArgumentParser(description='My capstone Design 2019')
    parses.add_argument("--roi", dest = 'roi', default = 0, help = "roi flag")
    parses.add_argument("--alpha", dest = 'alpha', default = 0, help = "center position add alpha")
    parses.add_argument("--video", dest = 'video', default = "drive3.mp4")
    parses.add_argument("--url", dest = 'url', default = False, type = str, help="youtube url link")
    parses.add_argument("--com", dest = 'com', default = False, help = "Setting Arduino port", type = str)
    parses.add_argument("--brate", dest = 'brate', default = 9600, help = "Setting Arduino baudrate")
    return parses.parse_args()

"""
python main.py --com COM4 --video drive.mp4
python main.py --com COM4 --roi 1 --video drive06.mp4 --alpha 60
python main.py --com COM4 --url https://youtu.be/YsPdvvixYfo --roi 1 --alpha 60
"""

args = arg_parse()

def save_video(filename, frame=60.0):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, frame, (1280,720))
    return out

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
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

def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame

    y_global_min = img.shape[0] # min은 y값 중 가장 큰값을 의미할 것이다. 또는 차로부터 멀어져 길을 따라 내려가는 지점 (?)
    y_max = img.shape[0]

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

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

        y_global_min = min(y1, y2, y_global_min)

    if (len(l_lane) == 0 or len(r_lane) == 0): # 오류 방지
        return 1

    l_slope_mean = np.mean(l_slope, axis =0)
    r_slope_mean = np.mean(r_slope, axis =0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1

    # y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    if np.isnan((y_global_min - l_b)/l_slope_mean) or \
    np.isnan((y_max - l_b)/l_slope_mean) or \
    np.isnan((y_global_min - r_b)/r_slope_mean) or \
    np.isnan((y_max - r_b)/r_slope_mean):
        return 1

    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

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

    global l_center
    global r_center
    global lane_center

    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    global uxhalf, uyhalf, dxhalf, dyhalf
    uxhalf = int((next_frame[2]+next_frame[6])/2)
    uyhalf = int((next_frame[3]+next_frame[7])/2)
    dxhalf = int((next_frame[0]+next_frame[4])/2)
    dyhalf = int((next_frame[1]+next_frame[5])/2)

    cache = next_frame

def lane_pts():
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_pts(flag=0):
    vertices1 = np.array([
                [230, 650],
                [620, 460],
                [670, 460],
                [1050, 650]
                ])

    vertices2 = np.array([
                [0, 720],
                [710, 400],
                [870, 400],
                [1280, 720]
    ])
    if flag is 0 : return vertices1
    if flag is 1 : return vertices2

hei = 25
alpha = int(args.alpha)
font_size = 1
""" 핸들 조종 및 위험 메세지 표시 """
def warning_text(image):
    whalf, height = 640, 720
    center = whalf - 5 + alpha
    angle = int(round(atan((dxhalf-(center))/120) * 180/np.pi, 3) * 3)

    m = 2
    limit = 0
    value = 0
    if angle > 90 : angle = 89
    if 90 > angle > limit :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Right', (150, hei*m), font, 0.8, red, font_size)
        value = angle

    if angle < -90 : angle = -89
    if -90 < angle < -limit:
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Left', (150, hei*m), font, 0.8, red, font_size)
        value = -angle + 100

    elif angle == 0 :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, white, font_size)
        cv2.putText(image, 'None', (150, hei*m), font, 0.8, white, font_size)
        value = 0

    # cv2.putText(image, 'angle = {0}'.format(angle), (10, hei*4), font, 0.7, white, font_size)
    # cv2.putText(image, 'value = {0}'.format(value), (10, hei*5), font, 0.7, white, font_size)

    if mcu_port:
        mcu.write([value])
    # print(value)

""" 현재 영상 프레임 표시 """
def show_fps(image, frames, start, color = white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f"%now_fps, (10, hei), font, 0.8, color, font_size)

""" Steering Wheel Control 시각화 """
def direction_line(image, height, whalf, color = yellow):
    cv2.line(image, (whalf-5+alpha, height), (whalf-5+alpha, 600), white, 2) # 방향 제어 기준선
    cv2.line(image, (whalf-5+alpha, height), (dxhalf, 600), red, 2) # 핸들 방향 제어
    cv2.circle(image, (whalf-5+alpha, height), 120, white, 2)

""" 왼쪽 차선, 오른쪽 차선, 그리고 차선의 중앙 지점 표시 """
def lane_position(image, gap = 20, length=20, thickness=2, color = red, bcolor = white): # length는 선의 위쪽 방향으로의 길이
    global l_cent, r_cent

    l_left = 300
    l_right = 520
    l_cent = int((l_left+l_right)/2)
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)

    r_left = 730
    r_right = 950
    r_cent = int((r_left+r_right)/2)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)

""" 왼쪽 차선과 오른쪽 차선을 직선으로 표시 """
def draw_lanes(image, thickness = 3, color = red):
    cv2.line(image, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), red, 3)
    cv2.line(image, (next_frame[6], next_frame[7]), (next_frame[4], next_frame[5]), red, 3)

""" 두 차선이 Cross하는 지점을 계산 """
def lane_cross_point():
    """
    y = m(x-a) + b (m은 negative)
    y = n(x-e) + f (n은 positive)
    -> x = (am - b - en + f)/(m-n)
    """
    for seq in range(8):
        if next_frame[seq] is 0: # next_frame 중 하나라도 0이 존재하면 break
            return (0, 0)
        else:
            l_slope = get_slope(next_frame[0], next_frame[1], next_frame[2], next_frame[3])
            r_slope = get_slope(next_frame[6], next_frame[7], next_frame[4], next_frame[5])

            x = (next_frame[0]*l_slope - next_frame[1] - next_frame[6]*r_slope + next_frame[7])/(l_slope-r_slope)
            y = l_slope*(x-next_frame[0]) + next_frame[1]
            return int(x), int(y)

""" 시점변경 """
def perspective(image): # Bird's eye view
    pts1 = np.float32([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[4], next_frame[5]], [next_frame[6], next_frame[7]]])
    pts2 = np.float32([[425, 0], [425, 720], [855, 0], [855, 720]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (1280, 720))
    cv2.line(dst, (l_cent, 0), (l_cent, 720), red, 2)
    cv2.line(dst, (r_cent, 0), (r_cent, 720), red, 2)
    return dst

""" 차선 검출을 위한 이미지 전처리 """
def process_image(image):
    global first_frame

    image = imresize(image, (720, 1280, 3))
    height, width = image.shape[:2]

    kernel_size = 3

    # Canny Edge Detection Threshold
    low_thresh = 150
    high_thresh = 200

    rho = 2
    theta = np.pi/180
    thresh = 100
    min_line_len = 50
    max_line_gap = 150

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow) # 흰색과 노란색의 영역을 합친다.
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw) # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(flag = 0)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    return result, roi_image

""" 차선 검출 결과물을 보여줌 """
def visualize(image, flg):
    height, width = image.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(image)
    vertices = [get_pts(flag=flg)]
    pts = lane_pts()

    gap = 25
    max = 100 # 410 ~ 760
    limit = 30
    if not lane_center[1] < hhalf:
        """ 차선검출 영역 최대 길이 이상 지정 """
        if r_center[0]-l_center[0] > max:
            cv2.fillPoly(zeros, [pts], lime)
            lane_position(zeros)
            direction_line(zeros, height = height, whalf = whalf)

    """ Lane Detection ROI """
    # cv2.putText(zeros, 'ROI', (930, 650), font, 0.8, yellow, font_size)
    # cv2.polylines(zeros, vertices, True, (0, 255, 255))
    return zeros

# object detection start and end point
""" 차량만 검출"""
def write(x, results, color = [126, 232, 229], font_color = red): # x = output
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1]) # 마지막 Index

    image = results
    label = "{0}".format(classes[cls])

    vals = [0, 2, 3, 5, 7] #not 2 or 3 or 5 or 7: # 인식할 Vehicles를 지정 (2car, 7truck, 5bus, 3motorbike)
    for val in vals:
        if cls == val:
            if not abs(c1[0]-c2[0]) > 1000: # 과도한 Boxing 제외
                centx = int((c1[0]+c2[0])/2)
                centy = int((c1[1]+c2[1])/2)

                if cls == 0:
                    cv2.rectangle(image, c1, c2, blue, 1)
                    t_size = cv2. getTextSize(label, font2, 1, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4
                    # cv2.rectangle(image, c1, c2, white, -1)
                    cv2.putText(image, label, (c1[0], c1[1] - t_size[1] + 4), font2, 1, blue, 1)

                else:
                    cv2.rectangle(image, c1, c2, red, 1) # 자동차 감지한 사각형
                    cv2.circle(image, (centx, centy), 3, blue, -1) # Detected vehicles' center

                    t_size = cv2.getTextSize(label, font2, 1, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4
                    # cv2.rectangle(image, c1, c2, white, -1)
                    cv2.putText(image, label, (c1[0], c1[1] - t_size[1] + 10), font2, 1, font_color, 1)
    return image

"""------------------------Data Directory------------------------------------"""
cfg = "cfg/yolov3.cfg"
weights = "weights/yolov3.weights"
names = "data/coco.names"

video_directory = "test_videos/"
args.video = "drive.mp4"
video = video_directory + args.video

# URL = "https://youtu.be/jieP4QkVze8"
# URL = "https://youtu.be/YsPdvvixYfo" roi = 1
# URL = ""

url = args.url
if url:
    vpafy = pafy.new(url)
    play = vpafy.getbest(preftype = "mp4")

"""--------------------------Changeable Variables----------------------------"""
frames = 0
first_frame = 1

start = 0
batch_size = 1
confidence = 0.8 # 신뢰도
nms_thesh = 0.4
resol = 640 # 해상도

num_classes = 12
print("Reading configure file")
model = Darknet(cfg)
print("Reading weights file")
model.load_weights(weights)
print("Reading classes file")
classes = load_classes(names)
set_requires_grad(model, False)
print("\nNetwork successfully loaded!")

mcu_port = args.com
mcu_brate = args.brate # Baud rate
if mcu_port:
    mcu = serial.Serial(mcu_port, mcu_brate, timeout = 1)
    mcu.timeout = None

model.net_info["height"] = resol
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

# clip1 = save_video('out_videos/detec_' + args.video, 30.0) # result 영상 저장
"""--------------------------Video test--------------------------------------"""
torch.cuda.empty_cache()

# CUDA = False
CUDA = torch.cuda.is_available()
if CUDA:
    model.cuda()
model.eval()

start = time.time()

if url:
    cap = cv2.VideoCapture(play.url)
else: # python main.py --com COM4 --youtub
    cap = cv2.VideoCapture(video)
print("\nVideo is now ready to show.")

while (cap.isOpened()):
    ret, frame = cap.read()
    # frame = imresize(frame, (720, 1280, 3))
    if ret:

        cv2.rectangle(frame, (0,0), (300, 130), dark, -1)

        show_fps(frame, frames, start, color = yellow)
        warning_text(frame)

        cpframe = frame.copy() # Lane frame copy

        """ Lane Detection """
        prc_img, _ = process_image(cpframe)
        lane_detection = visualize(prc_img, args.roi)

        """ Object Detection """
        if frames %3 == 0: # Frame 높히기 눈속임
            prep_frame = prep_image(frame, input_dim)
            frame_dim = frame.shape[1], frame.shape[0]
            frame_dim = torch.FloatTensor(frame_dim).repeat(1, 2)

            if CUDA:
                frame_dim = frame_dim.cuda()
                prep_frame = prep_frame.cuda()

                with torch.no_grad():
                    output = model(Variable(prep_frame, True), CUDA)
                    output = write_results(output, confidence, num_classes, nms_thesh)

                    frame_dim = frame_dim.repeat(output.size(0), 1)
            # scaling_factor = torch.min(416/frame_dim, 1)[0].view(-1, 1)
            scaling_factor = torch.min(resol/frame_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (input_dim - scaling_factor * frame_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (input_dim - scaling_factor * frame_dim[:, 1].view(-1, 1))/2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, frame_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, frame_dim[i,1])

            zero_frame = np.zeros_like(frame) # Object frame zero copy
            list(write(x, zero_frame) for x in output) # list(map(lambda x: write(x, frame), output))

            crossx, crossy = lane_cross_point()
            l_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (crossx, 0), (0, 0), (0, 720)])
            r_poly = Polygon([(next_frame[6], next_frame[7]), (crossx, crossy), (crossx, 0), (1280, 0), (1280, 720)])
            c_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (next_frame[6], next_frame[7])]) # Center Polygon

            cnt = 0 # Car count
            vals = [2, 3, 5, 7]
            l_cnt, r_cnt, c_cnt = 0, 0, 0
            for x in output:
                c1 = tuple(x[1:3].int())
                c2 = tuple(x[3:5].int())
                centx = int((c1[0]+c2[0])/2)
                centy = int((c1[1]+c2[1])/2)

                carbox = Polygon([(c1[0], c1[0]), (c1[0], c1[1]), (c1[1], c1[1]), (c1[1], c1[0])])
                carcent = Point((centx, centy)) # Car Center point

                """ 차의 중앙 지점과 겹치는 곳이 있으면 그곳이 차의 위치 """
                for val in vals:
                    if int(x[-1]) == val:
                        cnt += 1
                        if l_poly.intersects(carcent):
                            l_cnt += 1
                        if r_poly.intersects(carcent):
                            r_cnt += 1
                        if c_poly.intersects(carcent):
                            c_cnt += 1
                            if c_cnt > 1 : c_cnt = 1

                        if l_cnt or r_cnt or c_cnt:
                            cnt = l_cnt + c_cnt + r_cnt

            cv2.putText(frame, 'vehicles counting : {}'.format(cnt), (10, 75), font, 0.8, white, 1)
            cv2.putText(frame, 'L = {0} / F = {2} / R = {1}'.format(l_cnt, r_cnt, c_cnt), (10, 100), font, 0.7, white, font_size)

            object_result = cv2.add(frame, zero_frame)
            lane_result = cv2.addWeighted(object_result, 1, lane_detection, 0.5, 0)
            # lane_result = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)

            cv2.imshow("Result", lane_result)
        # clip1.write(lane_result)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
