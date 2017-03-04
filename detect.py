import cv2
import numpy as np

def nothing(x):
	pass

#img = np.zeros((400,400,3), np.uint8)
cap = cv2.VideoCapture(0)
# cv2.namedWindow('Controls')

# colors
AQUA = (255,255,0)
RED = (0,0,255)
WHITE = (255,255,255)

# Paint Board

paintBoard = np.ndarray((cap.get(4), cap.get(3), 3))
paintBoard[:,:,:] = WHITE
font = cv2.FONT_HERSHEY_SIMPLEX
print(cap.get(4), cap.get(3))

# cv2.createTrackbar('min H','Controls',0,180,nothing)
# cv2.createTrackbar('max H','Controls',0,180,nothing)
# cv2.createTrackbar('min S','Controls',0,255,nothing)
# cv2.createTrackbar('max S','Controls',0,255,nothing)
# cv2.createTrackbar('min V','Controls',0,255,nothing)
# cv2.createTrackbar('max V','Controls',0,255,nothing)

while True:

	ret, img = cap.read()
	img = cv2.flip(img,1)

	# nh = cv2.getTrackbarPos('min H','Controls')
	# ns = cv2.getTrackbarPos('min S','Controls')
	# nv = cv2.getTrackbarPos('min V','Controls')

	# xh = cv2.getTrackbarPos('max H','Controls')
	# xs = cv2.getTrackbarPos('max S','Controls')
	# xv = cv2.getTrackbarPos('max V','Controls')


	# img[:] = [b,g,r]

#	ret, mask = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
	
#	imgr = np.zeros(img.shape, np.uint8)
	imgx = np.ndarray(img.shape, np.uint8)
	imgx[:,:] = RED
#	imgr[:,:,2] = img[:,:,2]
#	img = cv2.bitwise_xor(imgx,img)
#	imgrgray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# lower_red = np.array([nh,ns,nv])
	# upper_red = np.array([xh,xs,xv])

	lower_orange = np.array([0,210,0])
	upper_orange = np.array([10,255,255])

	mask = cv2.inRange(hsv, lower_orange, upper_orange)
	img = cv2.bitwise_and(img,img, mask=mask)
#	cv2.imshow('Workspace',mask)
#	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,0)
#	_,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	gaus = cv2.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5), np.uint8)

#	opening = cv2.morphologyEx(gaus,cv2.MORPH_OPEN, kernel)
	dilation = cv2.dilate(gaus,kernel, iterations = 2)
	erosion = cv2.erode(dilation, kernel, iterations = 1)
	# cv2.imshow('Workspace 2',img)
	canny = cv2.Canny(gaus,100,300)
#	cv2.imshow('Workspace 3',canny)
	gaus = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)
	contours, heirarchy = cv2.findContours(gaus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	cv2.rectangle(paintBoard,(0,0),(int(cap.get(3)), int(cap.get(4))), WHITE , -1)
	if len(contours) > 0:
		for contour in contours:
#			cv2.rectangle(paintBoard,(0,0),(int(cap.get(4)), int(cap.get(3))), WHITE , 1)
			x,y,w,h = cv2.boundingRect(	contour)
			if w*h > 2000:
#				cv2.rectangle(canny, (x,y),(x+w,y+h),RED, 1)
#				cv2.rectangle(paintBoard, (x,y),(x+w,y+h),RED, 1)
				cv2.line(paintBoard, (x,y),(320,240),RED, 2)
				msg = 'Center'
				if x < 270:
					if y < 190:
						msg = 'Up Left'
					elif y > 290:
						msg = 'Down Left'
					else:
						msg = 'Left'
				elif x > 370:
					if y < 190:
						msg = 'Up Right'
					elif y > 290:
						msg = 'Down Right'
					else:
						msg = 'Right'
				else:
					if y < 190:
						msg = 'Up'
					elif y > 290:
						msg = 'Down'

				cv2.putText(paintBoard, msg, (220,320), font, 1, (0,0,0),3,cv2.CV_AA)
	cv2.imshow('Direction', paintBoard)
#	cv2.imshow('Edges', canny)
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()