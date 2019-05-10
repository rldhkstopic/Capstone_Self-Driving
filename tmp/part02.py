from moviepy.editor import VideoFileClip
from scipy.misc import imresize
import matplotlib.image as mpimg
import numpy as np
import cv2
import io
import os
import pickle

""" 참고 : 영상 해상도가 변할때마다 xm_per_pix, ym_per_pix를 고쳐주도록 하자"""
camera = pickle.load(open("pickle/camera_matrix.pkl", "rb"))
mtx = camera['mtx'] # Camera matrix
dist = camera['dist'] # Camera distortion matrix
camera_img_size = camera['imagesize']
src_points = np.float32([
[0, 690],
[500, 400],
[700, 400],
[1020, 720]
])

""" distortion process """
def distort_correct(img, mtx, dist, camera_img_size):
    img_size1 = (img.shape[1], img.shape[0])
    assert (img_size1 == camera_img_size), 'Image size is not compatible'
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

""" Sobel Gradient  """
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) # Rescale back to 8 bit integer
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

""" Gradient Magnitude : Gradient 크기 기준의 filtering """
def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # x와 yㅡ이 graident를 독립적으로 구한다.
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(x**2 + y**2) # Calculate the xy magnitude
    scale = np.max(mag)/255 # Scale to 8 bit(0-255)
    eightbit = (mag/scale).astype(np.uint8) # Convert type

    # mag threshold들이 만나는 지점에 binary mask를 만든다.
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1
    return binary_output

""" Graident Direction : Gradient의 기울기를 pi/2기준으로 filtering """
def dir_threshold(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output

""" Saturation Channel of HLS """
def hls_select(img, sthresh = (0, 255), lthresh = ()):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls_img[:, :, 1]
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1]) & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output

""" Red Channel of RGB """
def red_select(img, thresh = (0, 255)):
    R = img[:, :, 0]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output

""" Combining Filtr Methods : Yellow Lane Detection에 최적화된 Threshold 값 """
def binary_pipeline(img):
    img_copy = cv2.GaussianBlur(img, (3, 3), 0)

    # color channels
    s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
    #red_binary = red_select(img_copy, thresh=(200,255))

    # Sobel x
    x_binary = abs_sobel_thresh(img_copy, thresh=(25, 200))
    y_binary = abs_sobel_thresh(img_copy, thresh=(25, 200), orient='y')
    xy = cv2.bitwise_and(x_binary, y_binary)

    # magnitude & direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))

    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv2.bitwise_or(s_binary, gradient)

    return final_binary

""" Perspective Transform """
def warp_image(img, source_points):
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]

    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])

    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform(destination_points, source_points)

    warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)
    return warped_img, inverse_perspective_transform

""" Detecting Lane Lines """
def track_lanes_initialize(binary_warped):
    global window_search

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # 히스토그램 속 양 옆의 극대값을 얻기위한 분리과정
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    leftx_current = leftx_base
    rightx_current = rightx_base

    # the number of sliding windows : 잘못 지정하면 오류 발생
    nwindows = 9
    # Windows의 height 설정
    window_height = np.int(binary_warped.shape[0]/nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100 # Line의 양 옆 박스 Margin
    minpix = 50 # Recenter Windoew를 위한 최소 픽셀 수

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # x, y 그리고 좌우 경계선 선언
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3)

        # 윈도우에서 찾은 nonzero값들을 찾고 리스트에 추가한다.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # minpix보다 낮은 픽셀을 찾으면 다음 윈도우에서 평균 위치로 recenter한다.
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # 인덱스 배열을 모두 연결한다. (concatenate)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 좌우 픽셀 위치를 추출한다. (Extract)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit 2차 Polynomial output
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

""" Get Quadratic Polynomial Output """
def get_val(y, poly_coeff):
    return poly_coeff[0] * y ** 2 + poly_coeff[1] * y + poly_coeff[2]

""" Draw the Polynomials curves """
def lane_fill_poly(binary_warped, undist, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_val(ploty, left_fit)
    right_fitx = get_val(ploty, right_fit)

    # 차선을 그릴 새로운 이미지 생성
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y (The formula for radius of curvature)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp using inverse perspective transform
    newwarp = cv2.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0]))
    #newwarp = cv2.cv2tColor(newwarp, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

""" Lane Update """
def track_lanes_update(binary_warped, left_fit, right_fit):
    global window_search
    global frame_count

    if frame_count % 10 == 0:
        window_search=True

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit,right_fit,leftx,lefty,rightx,righty

""" Determine the lane curvature """
def measure_curve(binary_warped, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]) # y values
    y_eval = np.max(ploty)

    # 영상 해상도 1 픽셀당 meter단위 환산
    ym_per_pix = 30/720
    xm_per_pix = 3.7/1280

    # x positions lanes
    leftx = get_val(ploty, left_fit)
    rightx = get_val(ploty, right_fit)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # radius of curvature formula
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # 커브구간 차선의 중앙값 계산
    curve_rad = round((left_curverad + right_curverad)/2)
    return curve_rad

""" Determining where the vehicle is in the lane """
def vehicle_offset(img, left_fit, right_fit):
    xm_per_pix = 3.7/1280
    image_center = img.shape[1]/2

    # 차와 가장 가까운 이미지 하단에 선이 닿은 곳을 찾아내다.
    left_low = get_val(img.shape[0], left_fit)
    right_low = get_val(img.shape[0], right_fit)
    lane_center = (left_low+right_low)/2.0

    # vehicle offset
    distance = image_center - lane_center

    # convert to metric
    return (round(distance * xm_per_pix, 5))

def img_pipeline(img):
    global window_search
    global left_fit_prev
    global right_fit_prev
    global frame_count
    global curve_radius
    global offset

    img = imresize(img, (720, 1280, 3))
    undist = distort_correct(img, mtx, dist, camera_img_size)
    binary_img = binary_pipeline(undist) # get binary image
    birdseye, inverse_perspective_transform = warp_image(binary_img, src_points) #perspective transform

    if window_search: # False
        left_fit, right_fit = track_lanes_initialize(birdseye)

        left_fit_prev = left_fit
        right_fit_prev = right_fit

    else:
        left_fit = left_fit_prev
        right_fit = right_fit_prev

        left_fit, right_fit, leftx, lefty, rightx, righty = track_lanes_update(birdseye, left_fit,right_fit)

    left_fit_prev = left_fit
    right_fit_prev = right_fit

    processed_frame = lane_fill_poly(birdseye, undist, left_fit, right_fit) # draw polygon

    #update ~twice per second
    if frame_count==0 or frame_count%15==0:
        curve_radius = measure_curve(birdseye, left_fit, right_fit) # radii 계산
        offset = vehicle_offset(undist, left_fit, right_fit) # offset 계산

    font = cv2.FONT_HERSHEY_TRIPLEX
    processed_frame = cv2.putText(processed_frame, 'Radius: '+str(curve_radius)+' m', (30, 40), font, 1, (0,255,0), 2)
    processed_frame = cv2.putText(processed_frame, 'Offset: '+str(offset)+' m', (30, 80), font, 1, (0,255,0), 2)

    frame_count += 1
    return processed_frame

def save_video(filename):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(filename, fourcc, 60.0, (1280,720))
    return out

""" --------------------------------------------------------------------------------------"""
global frame_count
global window_search

window_search = True
frame_count = 0
video_name = '01.mp4'
image_name = '04.jpg'

image = mpimg.imread('test_images/'+image_name)
image = imresize(image, (720, 1280, 3))
image = distort_correct(image, mtx, dist, camera_img_size)

result = binary_pipeline(image)

birdeye_result, inverse_perspective_transform = warp_image(result, src_points)

cap = cv2.VideoCapture("test_videos/"+video_name)
clip1 = save_video('out_videos/adv_result_'+video_name) # result 영상 저장

while (cap.isOpened()):
    _, frame = cap.read()
    process_frame = img_pipeline(frame)
    # cv2.imshow("result", process_frame)
    clip1.write(process_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
image_name = '03.jpg'

image = mpimg.imread('test_images/'+image_name)
image = imresize(image, (720, 1280, 3))
image = distort_correct(image, mtx, dist, camera_img_size)

result = binary_pipeline(image)
birdeye_result, inverse_perspective_transform = warp_image(result)
# source_points = np.int32(source_points)
# destination_points = np.int32(destination_points)

# Detecting and Update all the Lanes on the road
left_fit, right_fit = track_lanes_initialize(birdeye_result)
left_fit, right_fit, leftx, lefty, rightx, righty = track_lanes_update(birdeye_result, left_fit, right_fit)
colored_lane = lane_fill_poly(birdeye_result, image, left_fit, right_fit)

# measure_curve(birdseye_result,left_fit, right_fit))
# offset = vehicle_offset(colored_lane, left_fit, right_fit)

plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

draw_poly1 = cv2.polylines(image, [source_points], True, (255, 0, 0), 5)

ax1.imshow(colored_lane)
ax1.set_title('colored_lane', fontsize=40)
ax2.imshow(birdeye_result, cmap='gray')
ax2.set_title('Destination', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.tight_layout()
plt.savefig('out_images/Filtering _images/sTransformed_'+image_name)

histogram = np.sum(birdeye_result[int(birdeye_result.shape[0]/2): , :], axis=0)
plt.figure()
plt.plot(histogram)
plt.savefig('out_images/Filtering _images/whiteHistogram_'+image_name)
"""
