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
import datetime

from utils import visualization_utils as vis_util
from utils import label_map_util

# Video Dimensions
# Width x Height: 640x480, 720x540, 1280x720
IM_WIDTH = 1280
IM_HEIGHT = 720
# Constants
MIN_CONFIDENCE_THRESHOLD = 0.60
AVG_CONFIDENCE_THRESHOLD = 0.71
FRAME_DETECT = 4
FRAME_BUFFER = 5

# Pathing and model setup
WIN_NAME = 'Team 4 - Car Count' 	# Name of Window
#VIDEO_NAME = 'BusOnly.mp4'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'  		# Name of the directory that contains the model to be used for prediction
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' 
LABELS = 'mscoco_label_map.pbtxt' 			# .pbtxt file with the labels
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
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables for entrance detection
entrance_buffer = 0
entrance_buffer_counter = 0
entrance_detector = False
entrance_detection_counter = 0
entrance_pause = 0
entrance_tl = (int(IM_WIDTH*0.7),int(IM_HEIGHT*0.05))
entrance_br = (int(IM_WIDTH*0.9),int(IM_HEIGHT*.95))
entrance_car_counts = 0

# Initialize variables for exit detection
exit_buffer = 0
exit_buffer_counter = 0
exit_detector = False
exit_detection_counter = 0
exit_pause = 0
exit_tl = (int(IM_WIDTH*0.5),int(IM_HEIGHT*0.05))
exit_br = (int(IM_WIDTH*0.6),int(IM_HEIGHT*.95))
exit_car_counts = 0

# Initialize other global variables
entry_id = 0
grab_vehicle = [0, 0]
grab_object_class = [0, 0]
vehicle_kind = ['N/A', 'N/A']

# Initialize session
sidfile = open("sessionidcount.txt", "r+")
session_id = int(sidfile.read())

#Initialize for RPi or Windows
#cap = cv2.VideoCapture(0)
#ret_val = cap.set(3,IM_WIDTH)
#ret_val = cap.set(4,IM_HEIGHT)

# Open video file
cap = cv2.VideoCapture(PATH_TO_VIDEO)

if cap.isOpened():
    window_handle = cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

def parking_detector(frame):

	# Set global variables
	global entrance_detector, exit_detector
	global entrance_buffer, entrance_buffer_counter, entrance_detection_counter, entrance_pause, entrance_car_counts
	global exit_buffer, exit_buffer_counter, exit_detection_counter, exit_pause, exit_car_counts
	global t, detection_time, entry_id, vehicle_kind, grab_vehicle, grab_object_class
	global exit_tl, exit_br

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
		min_score_thresh=MIN_CONFIDENCE_THRESHOLD)
		
	# Draw parking space
	cv2.rectangle(frame,entrance_tl,entrance_br,(20,255,20),3)
	cv2.putText(frame,"Entrance Line",(entrance_tl[0]+10,entrance_tl[1]-10),font,1,(255,20,255),2,cv2.LINE_AA)
	cv2.rectangle(frame,exit_tl,exit_br,(20,255,20),3)
	cv2.putText(frame,"Exit Line",(exit_tl[0]+10,exit_tl[1]-10),font,1,(255,20,255),2,cv2.LINE_AA)
	
	# Get class of detected object
	object_class = int(classes[0][0])
	object_scores = float(scores[0][0])
	objects_scores = np.squeeze(scores)
	objects_num = int(np.squeeze(num))
	
	# Get current time (from system)
	t = datetime.datetime.now()

	# Loop through all detected objects
	i = 0
	while i < objects_num:
	
		# Check to see if detected object is equal to or above the threshold
		# And check if detected object is a valid motor vehicle, (3 = car, 4 = bus, 6 = truck, 8 = motorcycle)
		object_class = int(classes[0][i])
		object_scores = float(scores[0][i])
		if((object_scores >= AVG_CONFIDENCE_THRESHOLD) and (object_class == 3 or object_class == 4 or object_class == 6 or object_class == 8)): #and (entrance_pause == 0)):
			
			# Get the center x and y position
			x = int(((boxes[0][i][1]+boxes[0][i][3])/2)*IM_WIDTH)
			y = int(((boxes[0][i][0]+boxes[0][i][2])/2)*IM_HEIGHT)
			cv2.circle(frame,(x,y), 5, (75,13,180), -1)			
			
			# If passes entrance, draw a circle at center of object, and increment entrance counter
			if ((x > entrance_tl[0]) and (x < entrance_br[0]) and (y > entrance_tl[1]) and (y < entrance_br[1])):
				entrance_detection_counter = entrance_detection_counter + 1
				entrance_buffer_counter = 0
				if grab_vehicle[0] == 0:
					grab_object_class[0] = object_class
					grab_vehicle[0] = 1
			
			# Do the same for exit area
			if ((x > exit_tl[0]) and (x < exit_br[0]) and (y > exit_tl[1]) and (y < exit_br[1])):
				#cv2.circle(frame,(x,y), 5, (75,13,180), -1)	
				exit_detection_counter = exit_detection_counter + 1
				exit_buffer_counter = 0			
				if grab_vehicle[1] == 0:
					grab_object_class[1] = object_class
					grab_vehicle[1] = 1
			
			
		i = i + 1
	
	# Once vehicle leaves entrance area, start buffering the counter
	if entrance_detection_counter > 0:
		entrance_buffer_counter = entrance_buffer_counter + 1
	if exit_detection_counter > 0:
		exit_buffer_counter = exit_buffer_counter + 1
	
	# If no vehicle is within the spot (false alarm), entrance_buffer by counting up to global frame buffer count
	if entrance_buffer_counter > FRAME_BUFFER:
		entrance_buffer_counter = 0
		entrance_detection_counter = 0
		entrance_pause = 0
		grab_vehicle[0] = 0
		entrance_detector = False
		
	# If vehicle is within a spot for more than global frame detect, set entrance_detector flag
	if ((entrance_detection_counter > FRAME_DETECT) and (entrance_pause == 0)):
		entrance_detector = True
		entrance_pause = 1	
		
	# If entrance_pause flag is set, draw message on screen. Count car and reset vars
	if entrance_pause == 1:
		if entrance_detector == True:
			entrance_car_counts = entrance_car_counts + 1
			entrance_buffer_counter = 0
			entrance_detection_counter = 0
			entrance_pause = 0
			entrance_detector = False
			detect_time = t.strftime("%H:%M:%S")
			datafile = open("carcount.txt", "a+")
			idfile = open("entryidcount.txt", "r+")
			entry_id = int(idfile.read())
			datafile.write("%d\t" % entry_id) #Unique key (for SQL)
			datafile.write("in\t") #In/Out
			datafile.write("%s\t" % detect_time) #Need to set entry_time when car occupies spot
			datafile.write("%s\t" % vehicle_kind[0]) #Vehicle type
			datafile.write("%d\t" % entrance_car_counts) #car count
			datafile.write("%d\n" % session_id) #session
			datafile.close()
			entry_id = entry_id + 1
			idfile.seek(0)
			idfile.truncate()
			idfile.write(str(entry_id))
			idfile.close()
		
	# Same concept as entrance but for exit
	if exit_buffer_counter > FRAME_BUFFER:
		exit_buffer_counter = 0
		exit_detection_counter = 0
		exit_pause = 0
		grab_vehicle[1] = 0
		exit_detector = False

	if ((exit_detection_counter > FRAME_DETECT) and (exit_pause == 0)):
		exit_detector = True
		exit_pause = 1	
	if exit_pause == 1:
		if exit_detector == True:
			exit_car_counts = exit_car_counts + 1
			exit_buffer_counter = 0
			exit_detection_counter = 0
			exit_pause = 0
			exit_detector = False
			detect_time = t.strftime("%H:%M:%S")
			datafile = open("carcount.txt", "a+")
			idfile = open("entryidcount.txt", "r+")
			entry_id = int(idfile.read())
			datafile.write("%d\t" % entry_id) #Unique key (for SQL)
			datafile.write("out\t") #In/Out
			datafile.write("%s\t" % detect_time) #Need to set entry_time when car occupies spot
			datafile.write("%s\t" % vehicle_kind[1]) #Vehicle type
			datafile.write("%d\t" % exit_car_counts) #car count
			datafile.write("%d\n" % session_id) #session
			datafile.close()
			entry_id = entry_id + 1
			idfile.seek(0)
			idfile.truncate()
			idfile.write(str(entry_id))
			idfile.close()
			
	if ((VIDEO_NAME == 'busandtruck_final2.mp4') and (exit_car_counts >= 1) and (entrance_car_counts >= 1) and exit_tl == (int(IM_WIDTH*0.67),int(IM_HEIGHT*0.3))):
		exit_tl = (int(IM_WIDTH*0.83),int(IM_HEIGHT*0.3))
		exit_br = (int(IM_WIDTH*0.94),int(IM_HEIGHT*.7))		
			
	i = 0
	while i < 2:
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
	cv2.putText(frame,'Entrance Detection counter: ' + str(entrance_detection_counter),(10,90),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Entrance Buffer counter: ' + str(entrance_buffer_counter),(10,110),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Entrance Pause: ' + str(entrance_pause),(10,130),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Entrance Buffer: ' + str(entrance_buffer),(10,150),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Exit Detection counter: ' + str(exit_detection_counter),(10,190),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Exit Buffer counter: ' + str(exit_buffer_counter),(10,210),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Exit Pause: ' + str(exit_pause),(10,230),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Exit Buffer: ' + str(exit_buffer),(10,250),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Entrance Cars counted: ' + str(entrance_car_counts),(10,290),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Exit Cars counted: ' + str(exit_car_counts),(10,310),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object kind: ' + str(vehicle_kind),(10,330),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object score: ' + str(object_scores),(10,350),font,0.5,(51,51,255),1,cv2.LINE_AA)
	cv2.putText(frame,'Object num: ' + str(objects_num),(10,370),font,0.5,(51,51,255),1,cv2.LINE_AA)

	return frame

# Run main program	
while cv2.getWindowProperty(WIN_NAME,0) >= 0:
	try:
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

		if cv2.waitKey(1) == ord('q'):
			break
		if cv2.waitKey(1) == ord('m'):
				shutil.move("carcount.txt", "/media/team4/ECS")
	except TypeError:
		print('TypeError occurred. If video reached end duration, ignore this.')
		break
	except AttributeError:
		print('AttributeError occurred. If video reached end duration, ignore this.')
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
