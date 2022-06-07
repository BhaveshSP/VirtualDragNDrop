import HandTrackingModule as htm 
import cv2


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
hand_detector = htm.HandDetector(detection_con=0.8)

cx,cy,w,h = 100,100,200,200
color_box = (233,22,0)
while True :
	_, img = cap.read()
	img = cv2.flip(img,1)
	all_hands = hand_detector.find_hands(img)
	if all_hands :
		hand = all_hands[0]
		lms_list = hand["lms"]
		index_finger = lms_list[8]
		# print(index_finger)
		middle_finger = lms_list[12]
		# print(middle_finger)
		distance = hand_detector.find_distance(index_finger,middle_finger,img)
		if cx-(w//2) < index_finger[0] < cx+(w//2) and cy-(h//2) < index_finger[1] < cy+(h//2):
			if distance < 20:
				color_box = (0,0,255)
				cx,cy = index_finger
			else:
				color_box = (233,22,0)

	
	cv2.rectangle(img,(cx-(w//2),cy-(h//2)),(cx+(w//2),cy+(h//2)),color_box,cv2.FILLED)
	cv2.imshow("Video Here",img)
	key = cv2.waitKey(1)
	if key == ord("s"):
		break

cap.release()
cv2.destroyAllWindows()



