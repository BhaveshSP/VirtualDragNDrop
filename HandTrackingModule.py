import cv2 
import mediapipe as mp 
import time 
import math 

class HandDetector:
	
	def __init__(self,mode=False,max_hands=2,complexity=1,detection_con=0.5,track_con=0.5):
		self.mode = mode 
		self.max_hands = max_hands 
		self.detection_con = detection_con 
		self.track_con = track_con 
		self.complexity = complexity
		self.mp_hands = mp.solutions.hands 
		self.hands = self.mp_hands.Hands(self.mode,self.max_hands,self.complexity,self.detection_con,self.track_con)
		self.mp_draw = mp.solutions.drawing_utils
	
	def find_hands(self,img,draw=True):

		img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(img_rgb)
		all_hands = [] 
		h,w,c = img.shape
		if self.results.multi_hand_landmarks:
			for hand_type, hand_lms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
				hand = {} 
				hand_lms_list = []
				x_list = [] 
				y_list = [] 
				for idx, lm in enumerate(hand_lms.landmark):
					x,y = int(lm.x*w), int(lm.y*h)
					x_list.append(x)
					y_list.append(y)
					hand_lms_list.append([x,y])


				x_min, x_max = min(x_list), max(x_list)
				y_min, y_max = min(y_list), max(y_list)
				box_width, box_height = x_max - x_min , y_max - y_min 
				hand_center = (x_min + (box_width // 2), y_min + (box_height//2))
				box_coord = x_min, y_min, box_width, box_height
				if draw:
					self.mp_draw.draw_landmarks(img,hand_lms,self.mp_hands.HAND_CONNECTIONS)
					cv2.rectangle(img,(box_coord[0]-20,box_coord[1]-20),
					              (box_coord[2]+box_coord[0]+20,box_coord[3]+box_coord[1]+20),(255,0,0),3)

				hand["lms"] = hand_lms_list
				hand["center"] = hand_center
				hand["box_coord"] = box_coord
				hand["type"] = hand_type.classification[0].label
				all_hands.append(hand)

		return all_hands
	def find_distance(self,point_one,point_two,img,draw=True):
		x_one,y_one = point_one
		x_two,y_two = point_two 
		distance = math.hypot(x_two-x_one,y_two-y_one)
		if draw:
			cv2.circle(img,point_one,10,(255,0,0),cv2.FILLED)
			cv2.circle(img,point_two,10,(255,0,0),cv2.FILLED)
			cv2.line(img,point_one,point_two,(255,0,0),2)
		return distance 


# Minimum Code for Running Module 

# def main():

# 	cap = cv2.VideoCapture(0)
# 	prev_time = 0 
# 	current_time = 0 
# 	hand_detector = HandDetector()

# 	while True :
# 		success, img = cap.read()
# 		current_time = time.time()
# 		img = cv2.flip(img,1)
# 		all_hands = hand_detector.find_hands(img)

# 		fps = 1 / (current_time - prev_time)
# 		prev_time = current_time
# 		cv2.putText(img,str(round(fps)),(10,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
		
# 		cv2.imshow("Video Here",img)
# 		key = cv2.waitKey(1)
# 		if key == ord("s"):
# 			break

# 	cap.release()
# 	cv2.destroyAllWindows()


# if __name__ == "__main__":
# 	main()