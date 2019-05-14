from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from keras.models import load_model
import numpy as np
import cv2

model = load_model('LaneModel.h5')
font = cv2.FONT_HERSHEY_TRIPLEX

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    global cache
    global first_frame
    det_slope = 0.5
    α = 0.2

    y_global_min = img.shape[0]
    y_max = img.shape[0]
    l_slope, l_lane = [], []
    r_slope, r_lane = [], []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)
        y_global_min = min(y1,y2,y_global_min)

    if((len(l_lane) == 0) or (len(r_lane) == 0)): # 오류 방지
        img = cv2.putText(img, 'No Lane Detected', (30, 40), font, 1, (0, 0, 255), 2)
        return 1
    else:
        img = cv2.putText(img, 'Successfully Lane Detected', (30, 40), font, 1, (255, 255, 255), 2)

    l_slope_mean = np.mean(l_slope, axis =0)
    r_slope_mean = np.mean(r_slope, axis =0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

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
        next_frame = (1-α) * prev_frame + α * current_frame

    margin = 10

    cv2.line(img, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), color, thickness)
    cv2.line(img, (next_frame[4], next_frame[5]), (next_frame[6], next_frame[7]), color, thickness)
    # cv2.polylines(img, [pts], True, (0, 255, 255), 2)

    cache = next_frame

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

""" Make ROI to minimize the detection error """
def region_of_interest(img, vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255, ) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

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

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    global first_frame

    kernel_size = 11

    # Canny Edget Detection Threshold
    low_thresh = 100
    high_thresh = 150

    rho = 3
    thresh = 100
    theta = np.pi/180
    min_line_len = 150
    max_line_gap = 180

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.

    lower_green = np.array([50, 128, 100], dtype = "uint8")
    upper_green = np.array([60, 255, 100], dtype = "uint8")
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    # lower_white = np.array([0, 0, 170], dtype = "uint8")
    # upper_white = np.array([255, 255, 255], dtype = "uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_white = cv2.inRange(gray_image, 200, 255)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow) # 흰색과 노란색의 영역을 합친다.
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw) # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출

    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)

    canny_edges = canny(gauss_gray, low_thresh, high_thresh)

    vertices = [get_pts(image)]
    roi_image = region_of_interest(canny_edges, vertices)

    line_image = hough_lines(roi_image, rho, theta, thresh, min_line_len, max_line_gap)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return result

def road_lines(image):
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = model.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)

    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = imresize(lane_drawn, (720, 1280, 3))
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result

def combination(deep, vision):


lanes = Lanes()
first_frame = 1

video_name = '01.mp4'
cap = cv2.VideoCapture("test_videos/" + video_name)

while (cap.isOpened()):
    _, frame = cap.read()
    frame = imresize(frame, (720, 1280, 3))

    result = road_lines(frame)
    process_frame = process_image(result)

    # cv2.imshow("result", result)
    cv2.imshow("result", process_frame)

    # cv2.imshow("lane_image", mask)
    #clip1.write(process_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# vid_output = "out_videos//proj_reg_vid.mp4"
#
# vid_clip = clip1.fl_image(road_lines, clip1)
# vid_clip.write_videofile(vid_output, audio=False)
