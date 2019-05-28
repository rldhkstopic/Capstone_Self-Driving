from __future__ import print_function

import cv2
import numpy as np
import video
import sys
import math

def angle(dx,dy):
    return math.atan2(dy,dx)*180/math.pi

def circle():
    a, b, c = circles.shape
    #if b>2:
    for i in range(b):
        cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
        print(i,"circle's BGR value is",cimg[int(circles[0][i][1]),int(circles[0][i][0])])

	print(cimg[int(circles[0][i][1]),int(circles[0][i][0])][0])

	circle_color_B=cimg[int(circles[0][i][1]),int(circles[0][i][0])][0]
	circle_color_G=cimg[int(circles[0][i][1]),int(circles[0][i][0])][1]
	circle_color_R=cimg[int(circles[0][i][1]),int(circles[0][i][0])][2]

	if circle_color_B<100 and circle_color_G<100 and circle_color_R>100 :
	   print("red light")

	if circle_color_B<100 and circle_color_G>100 and circle_color_R<100 :
	   print("green light")

	if circle_color_B<100 and circle_color_G>100 and circle_color_R>100 :
	   print("yellow light")

if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cap = video.create_capture(fn)
    while True:
        flag, src = cap.read()

        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(img, 50, 200)
        img = cv2.medianBlur(img, 5)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        cimg = src.copy() # numpy function
        row,col,ch=src.shape
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)

        if circles is not None:
            circle()
            
        cv2.imshow("detected circles", cimg)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
