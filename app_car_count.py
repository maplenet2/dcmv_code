# Using Pi Camera V2 on Jetson Nano
# Code reference to:
#
# Team 4
# Michael Khoo
#

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time

from utils import visualization_utils as vis_util
from utils import label_map_util

# Video Dimensions
# Width x Height: 640x480, 720x540, 1280x720
IM_WIDTH = 1280
IM_HEIGHT = 720

'''
# Setting up gstreamer pipeline for Pi Camera
def gstreamer_pipeline (
	capture_width=IM_WIDTH, 
	capture_height=IM_HEIGHT, 
	display_width=IM_WIDTH, 
	display_height=IM_HEIGHT, 
	framerate=60, 
	flip_method=0
):   
	return (
		'nvarguscamerasrc ! ' 
		'video/x-raw(memory:NVMM), '
		'width=(int)%d, height=(int)%d, '
		'format=(string)NV12, framerate=(fraction)%d/1 ! '
		'nvvidconv flip-method=%d ! '
		'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
		'videoconvert ! '
		'video/x-raw, format=(string)BGR ! appsink'
		% (
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height
		)
	)
'''
# Pathing and model setup
WIN_NAME = 'Car Detector Team 4' 	# Name of Window
VIDEO_NAME = 'count1.mp4'
MODEL_NAME = 'inference_graph' 		# Name of the directory that contains the model to be used for prediction
LABELS = 'labelmap.pbtxt' 			# .pbtxt file with the labels
NUM_CLASSES = 2 					# Number of classes in the identifier model
CWD_PATH = os.getcwd() 				# Get Directory
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb') # Path to the .pb file (model being used)
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', LABELS) # Path to file containing the tags mapped to object identifier
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME) # Path to video file

# Load the label mapping
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load Tensorflow model in memory
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    TFSess = tf.compat.v1.Session(graph=detection_graph)

# Tensors for the image, box detection, scores, classes, and number of objects
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize FPS counter
frame_rate_calc = 1
#frame_count = 0
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables for parking spot detection
buffer = 0
buffer_counter = 0
park_detector = False
park_counter = 0
pause = 0
pause_counter = 0
parking_spot_tl = (int(IM_WIDTH*0.5),int(IM_HEIGHT*0.05))
parking_spot_br = (int(IM_WIDTH*0.6),int(IM_HEIGHT*.95))
start_time = ''
end_time = ''
car_counts = 0

# Initialize Camera Section
# 	Initialize for Jetson Nano integrated camera (the pi-camera v2)
# 	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# 
#	Initialize for RPi or Windows
#	cap = cv2.VideoCapture(0)
#	ret_val = cap.set(3,IM_WIDTH)
#	ret_val = cap.set(4,IM_HEIGHT)

# Open video file
cap = cv2.VideoCapture(PATH_TO_VIDEO)

if cap.isOpened():
    window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

def parking_detector(frame):

	# Set global variables
	global park_detector
	global buffer, buffer_counter, park_counter, pause, pause_counter, car_counts
	global t, start_time, end_time

	# Read camera and grab RGB values of each pixel
	frame_expanded = np.expand_dims(frame, axis=0)

	# Perform object detection
	(boxes, scores, classes, num) = TFSess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	# Draw the results of on the video shown
	vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.atleast_2d(np.squeeze(boxes)),
		np.atleast_1d(np.squeeze(classes).astype(np.int32)),
		np.atleast_1d(np.squeeze(scores)),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.90)
		
	# Draw parking space
	cv2.rectangle(frame,parking_spot_tl,parking_spot_br,(20,255,20),3)
	cv2.putText(frame,"Imaginary Line",(parking_spot_tl[0]+10,parking_spot_tl[1]-10),font,1,(255,20,255),2,cv2.LINE_AA)
	
	# Get class of detected object
	object_class = int(classes[0][0])
	object_scores = float(scores[0][0])
	
	# Get current time (from system)
	t = time.localtime()
	
	# Check if detected object is a valid motor vehicle, (1 = Sedan, 2 = Truck, 3 = Bus)
	# Then get the center x and y position
	if((object_class == 1 or object_class == 2 or object_class == 3 or object_class == 4)): #and (pause == 0)):
		x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
		y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)		
		
		# If vehicle is in parking spot, draw a circle at center of object, and increment parking counter
		# Else if vehicle leaves spot, start buffering the counter
		if ((object_scores >= 0.95) and (x > parking_spot_tl[0]) and (x < parking_spot_br[0]) and (y > parking_spot_tl[1]) and (y < parking_spot_br[1])):
			cv2.circle(frame,(x,y), 5, (75,13,180), -1)	
			park_counter = park_counter + 1
			buffer_counter = 0
		elif park_counter > 0:
			buffer_counter = buffer_counter + 1

	# If no vehicle is within the spot (false alarm), buffer by counting up to 50 frames then reset counters
	if buffer_counter > 5:
		buffer_counter = 0
		park_counter = 0
		pause = 0
		pause_counter = 0
		park_detector = False
		end_time = time.strftime("%H:%M:%S", t)	

	# If vehicle is within a spot for more than 30 frames, set park_detector flag
	if ((park_counter > 5) and (pause == 0)):
		park_detector = True
		pause = 1	
		start_time = time.strftime("%H:%M:%S", t)		
		
	# If pause flag is set, draw message on screen
	if pause == 1:
		if park_detector == True:
			car_counts = car_counts + 1
			buffer_counter = 0
			park_counter = 0
			pause = 0
			pause_counter = 0
			park_detector = False
			#cv2.putText(frame,'Spot is occupied',(int(IM_WIDTH*.1),int(IM_HEIGHT*.5)),font,3,(0,0,0),7,cv2.LINE_AA)
			#cv2.putText(frame,'Car counted',(int(IM_WIDTH*.1),int(IM_HEIGHT*.85)),font,3,(95,176,23),5,cv2.LINE_AA)			
			#cv2.putText(frame,'Car parked at: ' + start_time,(int(IM_WIDTH*.5),190),font,1,(51,51,255),1,cv2.LINE_AA)
		
		# Increment pause counter until it reaches 50, then set pause flag back to 0
		#pause_counter = pause_counter + 1
		'''if pause_counter > 50:
			buffer = 0
			buffer_counter = 0
			pause = 0
			pause_counter = 0
			park_detector = False'''
			
			
			
	# Draw counter info
	cv2.putText(frame,'Detection counter: ' + str(park_counter),(10,90),font,0.5,(51,51,255),1,cv2.LINE_AA)
	#cv2.putText(frame,'Pause counter: ' + str(pause_counter),(10,110),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Buffer counter: ' + str(buffer_counter),(10,130),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Pause: ' + str(pause),(10,150),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Buffer: ' + str(buffer),(10,170),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Cars counted: ' + str(car_counts),(10,190),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object class: ' + str(object_class),(10,210),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object score: ' + str(object_scores),(10,230),font,0.5,(51,51,255),1,cv2.LINE_AA)

	return frame

# Run main program	
while cv2.getWindowProperty(WIN_NAME,0) >= 0:

	# Get initial tick value (used for calculating FPS)
	t1 = cv2.getTickCount()
	
	# Read each frame
	ret_val, frame = cap.read();
	frame.setflags(write=1)
	
	# Pass frame into detection function
	frame = parking_detector(frame)

	# Draw frames per second of the video
	cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

	# Show image with overlapping drawings
	cv2.imshow(WIN_NAME, frame)

	# Calculate FPS
	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frame_rate_calc = 1/time1

	# Debug
	#frame_count+=1
	#if frame_count == 3:
	#    break
	if cv2.waitKey(1) == ord('q'):
		break

# Cleanup video stream and windows
cap.release()
cv2.destroyAllWindows()
