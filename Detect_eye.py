from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor.dat",
	help="path to facial landmark predictor")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type=int, default=2,
	help="the number of consecutive frames the eye must be below the threshold")

args = vars(ap.parse_args())
EYE_AR_THRESH = args['threshold']
EYE_AR_CONSEC_FRAMES = args['frames']

COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
while cap.isOpened():
	ret, frame = cap.read()
	width, height = frame.shape[:2]
	whalf, hhalf = int(width/2), int(height/2)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	for rect in detector(gray, 0):
		shape = face_utils.shape_to_np(predictor(gray, rect))

		leftEye = shape[lStart:lEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEye = shape[rStart:rEnd]
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1, maxLevel=1)
		cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1, maxLevel=1)
		# Threshold만큼 눈이 감기면 Blink로 간주
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			if COUNTER > 60:
				cv2.putText(frame, "Dont Sleep", (whalf, hhalf), font, 1, (0, 0, 255), 2)
				print("Dont Sleep")
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			COUNTER = 0

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), font, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), font, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cap.stop()
cv2.destroyAllWindows()
