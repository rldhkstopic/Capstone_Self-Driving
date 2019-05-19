from __future__ import division
from torch.autograd import Variable
from torch.cuda import FloatTensor
from darknet import Darknet
from moviepy.editor import VideoFileClip
from scipy.misc import imresize
from PIL import Image

import numpy as np
import os
import io
import cv2
import math
import pafy
import pickle
import argparse
import time
import serial
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
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0))
next_frame = (0, 0, 0, 0, 0, 0, 0, 0)

def save_video(filename):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, 60.0, (1280,720))
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

    cache = next_frame

def get_lane_pts():
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
posgap = 0
font_size = 1
def warning_text(image): # Information Box
    l_pos = (l_cent - l_center[0])
    r_pos = (r_cent - r_center[0])

    if l_pos > 50 : l_pos = 50
    if r_pos > 50 : r_pos = 50
    if l_pos < -50 : l_pos = -50
    if r_pos < -50 : r_pos = -50

    m = 2
    limit = 20
    cv2.putText(image, 'l_pos = {0}'.format(l_pos), (10, hei*3), font, 0.7, white, font_size)
    cv2.putText(image, 'r_pos = {0}'.format(r_pos), (10, hei*4), font, 0.7, white, font_size)

    # cv2.putText(image, 'posgap = {0}'.format(posgap), (10, hei*5), font, 0.7, white, font_size)

    if l_pos > limit and -limit < r_pos :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Right', (160, hei*m), font, 0.8, red, font_size)
        value = int(((l_pos + r_pos) - 2*limit) / 2)
        # if mcu_port:
            # mcu.write([value])

    if r_pos > limit and -limit < l_pos :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Left', (160, hei*m), font, 0.8, red, font_size)
        value = int(((l_pos + r_pos) + 2*limit) / 2)
        value = -value + 100
        # if mcu_port:
            # mcu.write([value])

    else :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, white, font_size)
        cv2.putText(image, 'None', (160, hei*m), font, 0.8, white, font_size)
        value = 0
        # if mcu_port:
            # mcu.write([value])

    print(value)

def show_fps(image, frames, start, color = white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f"%now_fps, (10, hei), font, 0.8, color, font_size)

def direction_line(image, height, whalf, color = yellow):
    uxhalf = int((next_frame[2]+next_frame[6])/2)
    uyhalf = int((next_frame[3]+next_frame[7])/2)
    dxhalf = int((next_frame[0]+next_frame[4])/2)
    dyhalf = int((next_frame[1]+next_frame[5])/2)

    cv2.line(image, (whalf-5, height), (whalf-5, 600), white, 2) # 방향 제어 기준선
    cv2.line(image, (whalf-5, height), (dxhalf, 600), red, 2) # 핸들 방향 제어
    cv2.circle(image, (whalf-5, height), 120, white, 2)

""" 경고 기준선 : 이 선이 안전 기준선을 넘어가면 위험 """
def warning_baseline(image, height, whalf, color = yellow):
    cv2.line(image, (whalf, lane_center[1]), (lane_center[0], lane_center[1]), color, 2)

""" 안전 기준선 """
def safety_baseline(image, whalf, gap, length=10, color = white):
    cv2.line(image, (whalf-gap, lane_center[1]-length), (whalf-gap, lane_center[1]+length), color, 1)
    cv2.line(image, (whalf+gap, lane_center[1]-length), (whalf+gap, lane_center[1]+length), color, 1)

""" 왼쪽 차선, 오른쪽 차선, 그리고 차선의 중앙 지점 표시 """
def lane_position(image, gap = 20, length=20, thickness=2, color = red, bcolor = white): # length는 선의 위쪽 방향으로의 길이
    global l_cent, r_cent

    l_left = 300
    l_right = 520
    l_cent = int((l_left+l_right)/2)
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)

    cv2.line(image, (l_left, l_center[1]+length), (l_left, l_center[1]-length), bcolor, 1)
    cv2.line(image, (l_right, l_center[1]+length), (l_right, l_center[1]-length), bcolor, 1)
    cv2.line(image, (l_cent, l_center[1]+length-10), (l_cent, l_center[1]-length+10), bcolor, 1)
    cv2.line(image, (l_left, l_center[1]), (l_right, l_center[1]), bcolor, 1)

    r_left = 730
    r_right = 950
    r_cent = int((r_left+r_right)/2)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)

    cv2.line(image, (r_left, r_center[1]+length), (r_left, r_center[1]-length), bcolor, 1)
    cv2.line(image, (r_right, r_center[1]+length), (r_right, r_center[1]-length), bcolor, 1)
    cv2.line(image, (r_cent, r_center[1]+length-10), (r_cent, r_center[1]-length+10), bcolor, 1)
    cv2.line(image, (r_left, r_center[1]), (r_right, r_center[1]), bcolor, 1)

""" 왼쪽 차선과 오른쪽 차선을 직선으로 표시 """
def draw_lanes(image, thickness = 3, color = red):
    cv2.line(image, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), red, 3)
    cv2.line(image, (next_frame[6], next_frame[7]), (next_frame[4], next_frame[5]), red, 3)


""" Image processing to detect the lanes """
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

""" Visualize the information of lane detection """
def visualize(image, flg):
    height, width = image.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(image)
    vertices = [get_pts(flag=flg)]
    pts = get_lane_pts()

    gap = 25
    max = 100 # 410 ~ 760
    limit = 30
    whalf = int(width/2)
    hhalf = int(height/2)
    if not lane_center[1] < hhalf:
        """ 차선검출 영역 최대 길이 이상 지정 """
        if r_center[0]-l_center[0] > max:
            cv2.fillPoly(zeros, [pts], lime)
            lane_position(zeros)
            direction_line(zeros, height = height, whalf = whalf)

            # cv2.line(zeros, (dxhalf, dyhalf), (uxhalf, uyhalf), red, 2)

            # draw_lanes(zeros)

            # warning_baseline(zeros, height = height, whalf = whalf)
            # safety_baseline(frame, gap = gap, whalf = whalf)

    """ Lane Detection ROI """
    # cv2.putText(zeros, 'ROI', (930, 650), font, 0.8, yellow, font_size)
    # cv2.polylines(zeros, vertices, True, (0, 255, 255))

    return zeros

def frontcar(image):
    height, width = image.shape[:2]
    zeros = np.zeros_like(image)
    whalf = int(width/2)
    hhalf = int(height/2)

    gap = 15
    max = 100
    if not lane_center[1] < hhalf:
        if r_center[0]-l_center[0] > max:
            cv2.fillPoly(zeros, [get_lane_pts()], white)
            cv2.rectangle(zeros, (next_frame[0], 0), (next_frame[4], next_frame[5]), white, -1)
            # xor_zeros = cv2.bitwise_xor(limes, cent)
    return zeros

# object detection start and end point
""" Visualize the information of object detection """
def write(x, results, color = [126, 232, 229], font_color = red): # x = output
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1]) # 마지막 Index

    image = results
    label = "{0}".format(classes[cls])

    if cls == 2 or cls==3 or cls == 5 or cls == 7:#not 2 or 3 or 5 or 7: # 인식할 Vehicles를 지정 (2car, 7truck, 5bus, 3motorbike)
        if not abs(c1[0]-c2[0]) > 1000: # 과도한 Boxing 제외
            centx = int((c1[0]+c2[0])/2)
            centy = int((c1[1]+c2[1])/2)

            cv2.rectangle(image, c1, c2, red, 1) # 자동차 감지한 사각형
            cv2.circle(image, (centx, centy), 3, blue, -1) # Detected vehicles' center

            t_size = cv2.getTextSize(label, font2, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            # cv2.rectangle(image, c1, c2, white, -1)
            cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), font2, 1, font_color, 1)
    return image

def arg_parse():
    parses = argparse.ArgumentParser(description='My capstone Design 2019')
    parses.add_argument("--roi", dest = 'roi', default = 0, help = "roi flag")
    parses.add_argument("--video", dest = 'video', default = "drive3.mp4")
    parses.add_argument("--url", dest = 'url', default = False, type = str, help="youtube url link")
    parses.add_argument("--com", dest = 'com', default = False, help = "Setting Arduino port", type = str)
    parses.add_argument("--brate", dest = 'brate', default = 9600, help = "Setting Arduino baudrate")
    return parses.parse_args()

args = arg_parse()
"""------------------------Data Directory------------------------------------"""
cfg = "cfg/yolov3.cfg"
weights = "weights/yolov3.weights"
names = "data/coco.names"

video_directory = "test_videos/"
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
resol = 416 # 해상도

num_classes = 12
print("Reading configure file")
model = Darknet(cfg)
print("Reading weights file")
model.load_weights(weights)
print("Reading classes file")
classes = load_classes(names)
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

# clip1 = save_video('out_videos/lane_' + image_name) # result 영상 저장
"""--------------------------Video test--------------------------------------"""

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
        # prep_frame = prep_image(frame, input_dim)
        # frame_dim = frame.shape[1], frame.shape[0]
        # frame_dim = torch.FloatTensor(frame_dim).repeat(1, 2)
        #
        # if CUDA:
        #     frame_dim = frame_dim.cuda()
        #     prep_frame = prep_frame.cuda()
        #
        # with torch.no_grad():
        #     output = model(Variable(prep_frame, True), CUDA)
        # output = write_results(output, confidence, num_classes, nms_thesh)
        #
        # # if type(output) == int:
        # #     frames += 1
        # #     # cv2.imshow("Frame", frame)
        # #
        # #     key = cv2.waitKey(1)
        # #     if key & 0xFF == ord('q'):
        # #         break
        # #     continue
        #
        # frame_dim = frame_dim.repeat(output.size(0), 1)
        # scaling_factor = torch.min(416/frame_dim, 1)[0].view(-1, 1)
        #
        # output[:, [1, 3]] -= (input_dim - scaling_factor * frame_dim[:, 0].view(-1, 1))/2
        # output[:, [2, 4]] -= (input_dim - scaling_factor * frame_dim[:, 1].view(-1, 1))/2
        # output[:, 1:5] /= scaling_factor
        #
        # for i in range(output.shape[0]):
        #     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, frame_dim[i,0])
        #     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, frame_dim[i,1])
        #
        # zero_frame = np.zeros_like(frame) # Object frame zero copy
        # list(write(x, zero_frame) for x in output) # list(map(lambda x: write(x, frame), output))
        #
        # cnt = 0 # Car count
        # for x in output:
        #     if int(x[-1]) == 2 or int(x[-1]) == 3 or int(x[-1]) == 5 or int(x[-1]) == 7: cnt += 1
        # cv2.putText(frame, 'vehicles counting : {}'.format(cnt), (10, 75), font, 0.8, white, 1)
        #
        # object_result = cv2.add(frame, zero_frame)
        # lane_result = cv2.addWeighted(object_result, 1, lane_detection, 0.5, 0)
        lane_result = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)

        # cv2.imshow("rr", zeros)
        cv2.imshow("Result", lane_result)
        # clip1.write(lane_result)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
