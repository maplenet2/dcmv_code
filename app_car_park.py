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
import shutil
#from file_sys_helper import *

from utils import visualization_utils as vis_util
from utils import label_map_util

## CONSTANTS
# Video Dimensions
# Width x Height: 640x480, 720x540, 1280x720
IM_WIDTH = 1280
IM_HEIGHT = 720
# Constants
MIN_CONFIDENCE_THRESHOLD = 0.50
AVG_CONFIDENCE_THRESHOLD = 0.90

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
#VIDEO_NAME = 'test2-back.mp4'
VIDEO_NAME = 'BackTest2.mp4'
#VIDEO_NAME = 'test1-front.mp4'
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28' 		# Name of the directory that contains the model to be used for prediction, inference_graph
LABELS = 'mscoco_label_map.pbtxt' 			# .pbtxt file with the labels, labelmap.pbtxt
NUM_CLASSES = 80 					# Number of classes in the identifier model
CWD_PATH = os.getcwd() 				# Get Directory
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb') # Path to the .pb file (model being used)
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', LABELS) # Path to file containing the tags mapped to object identifier
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
buffer_counter = [0, 0, 0, 0]
car_count = [0, 0, 0, 0]
grab_vehicle = [0, 0, 0, 0]
grab_object_class = [0, 0, 0, 0]
park_detector = [False, False, False, False]
park_counter = [0, 0, 0, 0]
pause = [0, 0, 0, 0]
if VIDEO_NAME == 'test1-front.mp4':
	parking_spot_tl = [(int(IM_WIDTH*0.09),int(IM_HEIGHT*0.5)),(int(IM_WIDTH*0.38),int(IM_HEIGHT*0.5)),(int(IM_WIDTH*0.69),int(IM_HEIGHT*0.5))]
	parking_spot_br = [(int(IM_WIDTH*0.37),int(IM_HEIGHT*0.9)),(int(IM_WIDTH*0.68),int(IM_HEIGHT*0.9)),(int(IM_WIDTH*0.99),int(IM_HEIGHT*0.9))]
else:
	parking_spot_tl = [(int(IM_WIDTH*0.26),int(IM_HEIGHT*0.4)),(int(IM_WIDTH*0.44),int(IM_HEIGHT*0.4)),(int(IM_WIDTH*0.62),int(IM_HEIGHT*0.4))]
	parking_spot_br = [(int(IM_WIDTH*0.43),int(IM_HEIGHT*0.65)),(int(IM_WIDTH*0.61),int(IM_HEIGHT*0.65)),(int(IM_WIDTH*0.79),int(IM_HEIGHT*0.65))]
spot_count = len(parking_spot_tl)
start_time = ''
end_time = 'N/A'
vehicle_kind = ['N/A', 'N/A', 'N/A', 'N/A']
entry_id = 0

# Initialize session
sidfile = open("sessionid.txt", "r+")
session_id = int(sidfile.read())

# Initialize Camera Section
# 	Initialize for Jetson Nano integrated camera (the pi-camera v2)
#   cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# 
#	Initialize for RPi or Windows
#cap = cv2.VideoCapture(1)
#ret_val = cap.set(3,IM_WIDTH)
#ret_val = cap.set(4,IM_HEIGHT)

# Open video file
cap = cv2.VideoCapture(PATH_TO_VIDEO)

if cap.isOpened():
    window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

def parking_detector(frame):

	# Set global variables
	global park_detector
	global buffer_counter, car_count, park_counter, pause, entry_id, grab_vehicle, grab_object_class, session_id
	global start_time, end_time, vehicle_kind
	
	# Intialize variables
	x_coord = []
	y_coord = []
	vehicle_classes = []
	object_class = 0
	park_inspace = [False, False, False, False]
	
	# Read camera and grab RGB values of each pixel
	frame_expanded = np.expand_dims(frame, axis=0)

	# Perform object detection
	(boxes, scores, classes, num) = TFSess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	# Draw the results of on the video shown
	objects_boxes = np.squeeze(boxes)
	objects_classes = np.squeeze(classes).astype(np.int32)
	objects_scores = np.squeeze(scores)
	objects_num = int(np.squeeze(num))
	vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.atleast_2d(objects_boxes),
		np.atleast_1d(objects_classes),
		np.atleast_1d(objects_scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=MIN_CONFIDENCE_THRESHOLD)
		
	# Draw parking space
	i = 0
	while (i < len(parking_spot_tl)):
		cv2.rectangle(frame,parking_spot_tl[i],parking_spot_br[i],(255,20,20),3)
		cv2.putText(frame,"Parking Spot #1",(parking_spot_tl[i][0]+10,parking_spot_tl[i][1]-10),font,.5,(255,20,255),2,cv2.LINE_AA)
		i = i + 1
	
	# Get class of detected object
	object_scores = float(scores[0][0])
	
	# Get current time (from system)
	t = time.localtime()
	
	# Check if detected object is a valid motor vehicle, (1 = Sedan, 2 = Truck, 3 = Bus, 4 = Motorcycle)
	# Then get the center x and y position:		
	i = 0
	while i < objects_num:
		object_class = int(classes[0][i])
		if (float(scores[0][i]) > AVG_CONFIDENCE_THRESHOLD):
			if((object_class == 3 or object_class == 4 or object_class == 6 or object_class == 8)):
				x_temp = (int(((boxes[0][i][1]+boxes[0][i][3])/2)*IM_WIDTH))
				y_temp = (int(((boxes[0][i][0]+boxes[0][i][2])/2)*IM_HEIGHT))
				cv2.circle(frame,(x_temp,y_temp), 5, (75,13,180), -1)
				j = 0
				while(j < spot_count):
					if ((x_temp > parking_spot_tl[j][0]) and (x_temp < parking_spot_br[j][0]) and (y_temp > parking_spot_tl[j][1]) and (y_temp < parking_spot_br[j][1])):
						park_inspace[j] = True
						park_counter[j] = park_counter[j] + 1
						buffer_counter[j] = 0
						if grab_vehicle[j] == 0:
							grab_object_class[j] = object_class
							grab_vehicle[j] = 1
					j = j + 1
		i = i + 1
		
	i = 0
	while i < spot_count:
		if park_counter[i] > 0:
			buffer_counter[i] = buffer_counter[i] + 1

		# If no vehicle is within the spot (false alarm), buffer by counting up to 50 frames then reset counters
		if buffer_counter[i] > 50:
			buffer_counter[i] = 0		
			park_counter[i] = 0
			pause[i] = 0
			grab_vehicle[i] = 0
			park_detector[i] = False
			end_time = time.strftime("%H:%M:%S", t)
			datafile = open("datatext.txt", "a+")
			idfile = open("entryid.txt", "r+")
			entry_id = int(idfile.read())
			datafile.write("%d\t" % entry_id) #Unique key (for SQL)
			spot = i + 1
			datafile.write("%d\t" % spot) #Parking spot
			datafile.write("%s\t" % start_time) #Need to set entry_time when car occupies spot
			datafile.write("%s\t" % end_time) #Need to set exit time when car leaves spot
			datafile.write("%s\t" % vehicle_kind[i]) #Vehicle type
			car_count[i] = car_count[i] + 1
			datafile.write("%d\t" % car_count[i]) #car count
			datafile.write("%d\n" % session_id) #session
			datafile.close()
			entry_id = entry_id + 1
			idfile.seek(0)
			idfile.truncate()
			idfile.write(str(entry_id))
			idfile.close()

		# If vehicle is within a spot for more than 30 frames, set park_detector flag
		if ((park_counter[i] > 30) and (pause[i] == 0)):
			park_detector[i] = True
			pause[i] = 1	
			start_time = time.strftime("%H:%M:%S", t)	
			
		# If pause flag is set, draw message on screen
		if pause[i] == 1:
			if park_detector[i] == True:
				offset = int(20*i)
				spot_num = i + 1
				spot_name = str('Spot' + str(spot_num) + ' is occupied')
				spot_time = str('Car parked at: ' + str(start_time))
				spot_offset1 = int(190+offset)
				spot_offset2 = int(int(IM_HEIGHT*.85)+offset)
				cv2.putText(frame,spot_name,(int(IM_WIDTH*.1),spot_offset2),font,.75,(95,176,23),2,cv2.LINE_AA)			
				cv2.putText(frame,spot_time,(int(IM_WIDTH*.5),spot_offset1),font,.75,(51,51,255),2,cv2.LINE_AA)
			
		# Get name of object to be written in file
		if grab_object_class[i] == 3:
			vehicle_kind[i] = 'car'
		elif grab_object_class[i] == 4:
			vehicle_kind[i] = 'motorcycle'
		elif grab_object_class[i] == 6:
			vehicle_kind[i] = 'bus'
		elif grab_object_class[i] == 8:
			vehicle_kind[i] = 'truck'
		else:
			vehicle_kind[i] = 'Unknown'
		i = i + 1
		
	# Draw counter info
	cv2.putText(frame,'Detection counter: ' + str(park_counter),(10,90),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Buffer counter: ' + str(buffer_counter),(10,110),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Pause: ' + str(pause),(10,130),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Last car parked: ' + end_time,(10,150),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Last vehicle: ' + str(vehicle_kind),(10,170),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Grab object class: ' + str(grab_object_class),(10,190),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object class: ' + str(object_class),(10,210),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object score: ' + str(object_scores),(10,230),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Objects dectected: ' + str(objects_num),(10,250),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Vehicles in Frame: ' + str(park_inspace),(10,270),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Veh x-coord: ' + str(x_coord),(10,290),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Veh y-coord: ' + str(y_coord),(10,310),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Session: ' + str(session_id),(10,330),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Car count: ' + str(car_count),(10,350),font,0.5,(51,51,255),1,cv2.LINE_AA)
	
	return frame

# Run main program	
while cv2.getWindowProperty(WIN_NAME,0) >= 0:
	try:
		# Get initial tick value (used for calculating FPS)
		t1 = cv2.getTickCount()
		
		# Read each frame
		ret_val, frame = cap.read();
		#frame.setflags(write=1)
		
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
		if cv2.waitKey(1) == ord('m'):
			shutil.move("datatext.txt", "/media/team4/ECS")
	except TypeError:
		print('TypeError occurred. If video reached end duration, ignore this.')
		break

# Write to session file
session_id = session_id + 1
sidfile.seek(0)
sidfile.truncate()
sidfile.write(str(session_id))
sidfile.close()
		
# Cleanup video stream and windows
cap.release()
cv2.destroyAllWindows()
