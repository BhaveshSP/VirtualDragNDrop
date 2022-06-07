import HandTrackingModule as htm 
import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
hand_detector = htm.HandDetector(detection_con=0.8)

class DragBox:
	def __init__(self,center,size=[80,80]):
		self.center = center 
		self.size = size 

	def update(self,cursor):
		cx,cy = self.center 
		if cx-(w//2) < cursor[0] < cx+(w//2) and cy-(h//2) < cursor[1] < cy+(h//2):
			self.center = cursor



box_list = [] 
for i in range(4):
	box_list.append(DragBox([i*100+100,100]))

while True :
	_, img = cap.read()
	img = cv2.flip(img,1)
	all_hands = hand_detector.find_hands(img,draw=False)
	if all_hands :
		hand = all_hands[0]
		lms_list = hand["lms"]
		index_finger = lms_list[8]
		# print(index_finger)
		middle_finger = lms_list[12]
		# print(middle_finger)
		distance = hand_detector.find_distance(index_finger,middle_finger,img)
		if distance < 30:
			for box in box_list:
				box.update(index_finger)
	new_img = np.zeros_like(img,np.uint8)
	for box in box_list:
		cx,cy = box.center
		w,h = box.size

		cv2.rectangle(new_img,(cx-(w//2),cy-(h//2)),(cx+(w//2),cy+(h//2)),(0,220,0),cv2.FILLED)

	out = img.copy()
	alpha = 0.1
	mask = new_img.astype(bool)
	out[mask] = cv2.addWeighted(img,alpha,new_img,1-alpha,0)[mask]
	cv2.imshow("Video Here",out)

	key = cv2.waitKey(1)
	if key == ord("s"):
		break

cap.release()
cv2.destroyAllWindows()



