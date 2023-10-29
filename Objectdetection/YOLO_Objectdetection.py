#Object-Detection based on a video frame

import random
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from Objectdetection.tracker import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import supervision as sv
import tkinter as tk
from PIL import Image, ImageTk
from Middlepart.AtmoApi import config


# Objects and values detected will be stored in dict
Showframe = 0
byte_tracker = sv.ByteTrack()
byte_tracker2 = sv.ByteTrack()
annotator = sv.BoxAnnotator()
annotator2 = sv.BoxAnnotator()
thisdict = {}
FrameNbr=0
fps = 0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.dropout0 = nn.Dropout(p=0.7)
        self.fc0 = nn.Linear(32 * 28 * 51, 128)
        self.fc0_bn = nn.BatchNorm1d(32 * 28 * 51, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.max_pool2d(F.elu(self.conv1_bn(self.conv1(x))),2)
        x = F.max_pool2d(F.elu(self.conv2_bn(self.conv2(x))),2)
        x = F.max_pool2d(F.elu(self.conv3_bn(self.conv3(x))),2)
        x = x.view(-1, 32 * 28 * 51)
        x = F.elu(self.fc0(self.dropout0(self.fc0_bn(x))))
        x = F.elu(self.fc1(self.dropout1(x)))
        x = F.elu(self.fc2(x))
        return x

def fill_dictionary(x, y, w, h, FrameNbr, fps, class_list, clsID, id):
    """
    Update or create a dictionary with information about bounding boxes of objects of the video frame.

    Args:
        x (float): The x-coordinate of the top-left corner of the bounding box.
        y (float): The y-coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        FrameNbr (int): The number of the current video frame.
        fps (float): The frames per second (FPS) of the video.
        class_list (list): A list of class names or labels.
        clsID (int): The index number of the object class in the class_list.
        id (int): A unique identifier for the detected and tracked object.

    Returns:
        dict: The updated dictionary containing object information for the audio-part.

    Description:
        This function is used to maintain a dictionary that stores information about object bounding boxes in video frames.
        It calculates the center (x_center, y_center) of the bounding box, as well as the actual width and height of the box. It also calculates the time (t) at which the frame was captured in the video.
        If the 'id' already exists in the dictionary, it appends the new data to the existing lists for 'x', 'y', 'w', 'h', and 't'. If the 'id' does not exist, a new entry is created in the dictionary with all the relevant information, including the object's class.
        The updated dictionary is returned after the addition or modification of object data.
    """

    class_type = class_list[int(clsID)]
    # w and h are x and y values of lower right boundingbox-border 
    y = - y
    h = - h
    width = w - x # calculate actual width of bounding box
    height = y - h # calculate actual height of bounding box
    x_center = x + width / 2
    y_center = y - height / 2
    t = FrameNbr / fps

    if id in thisdict: 
        thisdict[id]["x"].append(x_center)
        thisdict[id]["y"].append(y_center)
        thisdict[id]["w"].append(width)
        thisdict[id]["h"].append(height)
        thisdict[id]["t"].append(t)

    else: 
        thisdict[id] = {"object class": class_type,
                        "x": [x_center], 
                        "y": [y_center], 
                        "w": [width], 
                        "h": [height], 
                        "t": [t] }
        
    return thisdict

def analyze_scene_from_video(frame, class_list_scene):
    """
    Analyze the scene in a video frame using a YOLO-based object detection model.

    Args:
        frame (Image): The video frame to analyze.
        class_list_scene (list): A list of class names or labels for scene categories.

    Returns:
        int: The predicted scene category index.

    Description:
        This function is used to analyze the scene in a given video frame using a YOLO-based object detection model.
        
        - It loads the YOLO model for scene analysis with the specified weights.
        - It makes predictions on the input frame using the model.
        - It extracts the probabilities of different scene categories.
        - It returns the index of the most likely scene category.

        The `class_list_scene` parameter is expected to contain a list of scene category names or labels that correspond to the model's output.
    """
    # print("Scene Analyse")
    modelScene = YOLO(os.path.join("Objectdetection","weights","best_stadt_sand_wald.pt"))
    results = modelScene.predict(frame)   
    probs = results[0].probs
    # print(class_list_scene)
    vorhersage = int(probs.top1)
    
    return vorhersage
    # print("Die Szene ist im:" + str(class_list_scene[int(probs.top1)]))
    
def load_model_tracker(cap):
    """
    Initialize an object tracker and load YOLO models and configuration for object detection.

    Args:
        cap: Video capture object to get video properties.

    Returns:
        tuple: A tuple containing various initialized objects and data, including the tracker, class lists,
               YOLO models, detection colors, frame dimensions, and video properties.

    Description:
        This function initializes an object tracker for tracking objects in video frames and loads YOLO models
        and related configuration data. It performs the following tasks:

        1. Initializes the Euclidean Distance Tracker (`tracker`) for object tracking.
        2. Reads the COCO class list from a file and stores it in `class_list`.
        3. Reads the scene category list from another file and stores it in `class_list_scene`.
        4. Loads two pretrained YOLO models for object detection: `model` and `model2`.
        5. Calculates the video length and frame rate based on the provided video capture object `cap`.
        6. Generates random colors for class list items and stores them in `detection_colors`.
        7. Retrieves frame dimensions (`frame_wid` and `frame_hyt`) from a configuration dictionary (not shown).
        8. Determines the video frame rate while handling different OpenCV versions.
        9. Returns a tuple containing all the initialized variables and objects for later use in the code.
    """
    
    tracker = EuclideanDistTracker()
    my_file = open("Objectdetection/utils/coco.txt", "r")
    
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()
    scene_List = open(os.path.join("Objectdetection","utils","SceneListCoco.txt"), "r")
    data_scene = scene_List.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list_scene = data_scene.split("\n")
    scene_List.close()
    
    # load a pretrained model
    model = YOLO(os.path.join("Objectdetection","weights","yolov8n.pt"), "v8")
    model2 = YOLO(os.path.join("Objectdetection","weights","ocean_model.pt"), "v8")
    
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/int(cap.get(cv2.CAP_PROP_FPS))
    # Generate random colors for class list
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))        


    frame_wid = config["frame_width"]
    frame_hyt = config["frame_height"]  
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        # print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # print("Die Framerate ist:", fps)
    return tracker, class_list, class_list_scene, model, model2, detection_colors, frame_wid, frame_hyt, fps, video_length
    
def analyse_frame_object(tracker, class_list, class_list_scene, model,  model2, detection_colors, frame_wid, frame_hyt, fps, frame, FrameNbr):    
    """
    Analyze objects in a video frame, perform object detection and tracking, and update object information.

    Args:
        tracker: Object tracker for tracking objects across frames.
        class_list: List of class names or labels for object detection.
        class_list_scene: List of scene category names or labels.
        model: YOLO object detection model.
        model2: Another YOLO object detection model.
        detection_colors: Colors for visualizing detected objects.
        frame_wid: Width of the video frame.
        frame_hyt: Height of the video frame.
        fps: Frames per second of the video.
        frame: The current video frame.
        FrameNbr: The frame number in the video.

    Returns:
        thisDict: The dict containing information about the boundingboxes for the audio part
        frame: the processed frame 

    Description:
        This function is responsible for analyzing objects within a given video frame. It performs the following tasks:

        1. Resize the input frame to a specified width and height.
        2. Perform object detection using the YOLO models (`model` and `model2`) on the resized frame with confidence thresholds.
        3. Retrieve object detection parameters, including bounding boxes and class IDs.
        4. Calculate scale factors for resizing bounding boxes according to the frame dimensions.
        5. Process and draw bounding boxes for detected objects on the frame.
        6. Update the object information dictionary (`thisdict`) with object positions, classes, and frame information.
        7. Return the updated object information dictionary and the processed frame.
        The `tracker` is used for object tracking across frames, and the `class_list_scene` is used to analyze scene categories.
    """
    frame = cv2.resize(frame, (frame_wid, frame_hyt))
        
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.75, save=False)
    detect_params2 = model2.predict(source=[frame], conf=0.80, save=False)
    
    '''    
    if FrameNbr == 1 or FrameNbr % 5 == 0:
        vorhersage = analyze_scene_from_video(frame, class_list_scene)
      # Convert tensor array to numpy
    '''

    DP = detect_params[0].numpy()   
    DP2 = detect_params2[0].numpy()
    
    scfx, scfy = get_scalefactor(frame_wid, frame_hyt)
    
      
    detections = []
    if len(DP2) != 0:
        for i in range(len(detect_params2[0])):
            boxes = detect_params2[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            clsID = clsID + 80

            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
                
            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2])
            h = int(bb[3])
                
            detections.append([x,y,w,h,clsID])
            nebbx = resize_bbox([x,y,w,h], scfx, scfy)
            
            cv2.rectangle(
                frame,
                (int(nebbx[0]), int(nebbx[1])),
                (int(nebbx[2]), int(nebbx[3])),
                detection_colors[int(clsID)],
                3,
                )  
                 
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
                ## print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2])
            h = int(bb[3])
                
            detections.append([x,y,w,h,clsID])

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
                )       
                 
    #Object-Tracking
    boxes_ids = tracker.update(detections)
  
    for box_id in boxes_ids:        
        x,y,w,h, id, clsID = box_id
        # print(clsID , "clsID")
        fill_dictionary(x, y, w, h, FrameNbr, fps, class_list, clsID, id)
        cv2.putText(frame, str(id) +"  "+ class_list[int(clsID)], (x-15, y -15), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        #datei.write("\r\n" +"Objektnbr:    "+ str(id) +"      :     "+ str(class_list[int(clsID)]))
        #datei.write("\r\n" + "x: "+ str(x) + "y: "+ str(y))
   
    return thisdict, frame

def fill_dictionary2(x, y, w, h, FrameNbr, fps, class_type, id):
    """
    Update object information in a dictionary with data from a video frame.

    Args:
        thisdict (dict): A dictionary storing object information.
        class_type (str): The class or category of the object.
        x (float): The x-coordinate of the object's bounding box.
        y (float): The y-coordinate of the object's bounding box.
        w (float): The width of the object's bounding box.
        h (float): The height of the object's bounding box.
        FrameNbr (int): The frame number in the video.
        fps (float): The frames per second of the video.
        id (int): A unique identifier for the object.

    Returns:
        dict: The updated dictionary with object information.

    Description:
        This method is used to update a dictionary ('thisdict') with information about objects detected in video frames. It performs the following tasks:

        - Adjusts the 'y' and 'h'.
        - Calculates the actual width and height of the bounding box based on the coordinates of its corners.
        - Computes the center coordinates of the bounding box.
        - Calculates the time 't' at which the frame was captured in the video.
        - Checks if the object with the given 'id' already exists in the dictionary.
        - If the object exists, it appends the new data (x, y, width, height, and time) to existing lists.
        - If the object doesn't exist, it creates a new entry in the dictionary with object information.

        The updated dictionary, including the new object information, is then returned.
    """
    

    class_type = class_type
    y = - y
    h = - h
    # w and h are x and y values of lower right boundingbox-border 
    width = w - x # calculate actual width of bounding box
    height = y - h # calculate actual height of bounding box
    x_center = x + width / 2
    y_center = y - height / 2
    
    # # # print(fps)
    # # # print(FrameNbr)
    
    t = FrameNbr / fps
    
    

    if id in thisdict: 
        thisdict[id]["x"].append(x_center)
        thisdict[id]["y"].append(y_center)
        thisdict[id]["w"].append(width)
        thisdict[id]["h"].append(height)
        thisdict[id]["t"].append(t)

    else: 
        thisdict[id] = {"object class": class_type,
                        "x": [x_center], 
                        "y": [y_center], 
                        "w": [width], 
                        "h": [height], 
                        "t": [t] }
        
    # # # print(thisdict)

def callback(frame: np.ndarray, index: int, model, model2, FrameNbr, fps) -> np.ndarray:
    """
    Process a video frame, perform object detection and tracking, and return an annotated frame and updated object information.

    Args:
        frame (np.ndarray): A video frame as a NumPy array.
        index (int): The index of the current frame.
        model: The first object detection model.
        model2: The second object detection model (custom).
        FrameNbr (int): The frame number in the video.
        fps (float): The frames per second of the video.

    Returns:
        np.ndarray: The annotated video frame with detected objects.
        dict: The updated dictionary containing object information.

    Description:
        This method processes a video frame, performs object detection using two YOLO models (model and model2),
        and combines the results into a single annotated frame with labels. It also updates a dictionary ('thisdict')
        with object information for the detected objects.

        The process involves the following steps:
        1. Resize the input frame to match the desired dimensions defined in the 'config'.
        2. Perform object detection using both YOLO models.
        3. Filter detections based on a minimum confidence threshold defined in the 'config'.
        4. Update object tracking using the detected objects.
        5. Annotate the video frame with bounding boxes and labels for detected objects.
        6. Extract and update object information for each detected object and store it in 'thisdict'.
        7. Return the annotated frame and the updated object information dictionary.
    """
    

    frame_wid = config["frame_width"]
    frame_hyt = config["frame_height"]
    min_confidence_threshold = config["min_confidence_threshold"]
    
    frame = cv2.resize(frame, (frame_wid, frame_hyt))
            
    results = model(frame)[0]
    results2 = model2(frame)[0]
    
    detections = sv.Detections.from_ultralytics(results)
    detections2 = sv.Detections.from_ultralytics(results2)
    
    detections = detections[detections.confidence >  min_confidence_threshold]
    detections2 = detections2[detections2.confidence >  min_confidence_threshold]    
    
    detections = byte_tracker.update_with_detections(detections)
    detections2 = byte_tracker2.update_with_detections(detections2)
    
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.3f}"
        for _, _, confidence, class_id, tracker_id in detections
    ]
    
    labels2 = [
        f"#{tracker_id} {model2.model.names[class_id]} {confidence:0.3f}"
        for _, _, confidence, class_id, tracker_id in detections2
    ]
    
    
    
    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
    new_annotated_frame = annotator2.annotate(scene=annotated_frame.copy(), detections=detections2, labels=labels2)
    

    for items in detections:
        position, _ , confidence, class_id, tracker_id = items    
        # # print(position, "position")
        class_type = model.model.names[class_id]
        
        x = position[0]
        y = position[1]
        w = position[2]
        h = position[3]
            
        fill_dictionary2(x, y, w, h, FrameNbr, fps, class_type, tracker_id)
        
    for items in detections2:
        position, _ , confidence, class_id, tracker_id = items         
        # print(position, "position")
        class_type = model2.model.names[class_id]
        
        x = position[0]
        y = position[1]
        w = position[2]
        h = position[3]
            
        fill_dictionary2(x, y, w, h, FrameNbr, fps, class_type, tracker_id)

    return new_annotated_frame, thisdict 

def load_models(cap):
    """
    Initialize object detection models, object tracking trackers, annotator, and video properties.

    Args:
        cap: Video capture object for accessing video properties.

    Returns:
        model: The first YOLO object detection model.
        model2: The second YOLO object detection model (custom).
        byte_tracker: ByteTrack object for object tracking with the first model.
        byte_tracker2: ByteTrack object for object tracking with the second model.
        annotator: BoxAnnotator object for annotating detected objects.
        fps: Frames per second of the video.
        video_length: Total length of the video in seconds.

    Description:
        This method initializes various components for processing video frames, including:
        - Two YOLO object detection models: 'model' and 'model2'.
        - Two ByteTrack objects for object tracking: 'byte_tracker' and 'byte_tracker2'.
        - A BoxAnnotator object for annotating detected objects: 'annotator'.
        - Determines the frames per second (FPS) of the video and video length based on OpenCV version.
        - Returns all the initialized objects and video properties for further processing.
        
    """
    model = YOLO("Objectdetection/weights/yolov8n.pt")
    model2 = YOLO("Objectdetection/weights/ocean_model.pt")
    byte_tracker = sv.ByteTrack()
    byte_tracker2 = sv.ByteTrack()
    annotator = sv.BoxAnnotator()  
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)      
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Die Framerate ist:", fps)
        
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/int(cap.get(cv2.CAP_PROP_FPS))  
    
    return model, model2, byte_tracker,byte_tracker2, annotator, fps, video_length

def resize_bbox(bbox, scale_factor_x, scale_factor_y):    
    """
    Resize a bounding box (bbox) by scaling its dimensions with given scale factors.

    Args:
        bbox (list): A list representing a bounding box in the format [x1, y1, x2, y2].
        scale_factor_x (float): Scale factor for the width (along the x-axis).
        scale_factor_y (float): Scale factor for the height (along the y-axis).

    Returns:
        list: A resized bounding box in the format [new_x1, new_y1, new_x2, new_y2].

    Description:
        This method takes a bounding box 'bbox' and resizes it by scaling its dimensions along the x and y axes.
        The scaling factors 'scale_factor_x' and 'scale_factor_y' are applied to the width and height of the bounding box.

        The bounding box is represented as [x1, y1, x2, y2], where (x1, y1) is the top-left corner, and (x2, y2) is the
        bottom-right corner. The method calculates the new dimensions, new_x1, new_y1, new_x2, and new_y2, and returns
        the resized bounding box.

        Note: The returned values are rounded to integers to obtain valid pixel coordinates.
    """
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    cx = x1 + width / 2
    cy = y1 + height / 2
    
    new_width = width * scale_factor_x
    new_height = height * scale_factor_y
  
    new_x1 = int(cx - new_width / 2)
    new_y1 = int(cy - new_height / 2)
    new_x2 = int(cx + new_width / 2)
    new_y2 = int(cy + new_height / 2)


    return [new_x1, new_y1, new_x2, new_y2]

def get_scalefactor(framewidth, frameheight):   
    """
    Calculate scaling factors for adjusting the frame size to fit within a specified window size in a GUI.

    Args:
        framewidth (int): The width of the video frame.
        frameheight (int): The height of the video frame.

    Returns:
        tuple: A tuple containing two scaling factors (scfx, scfy) for the x-axis and y-axis, respectively.

    Description:
        This method calculates scaling factors to adjust the size of a video frame to fit within a specified window size
        in a graphical user interface (GUI). The 'framewidth' and 'frameheight' parameters represent the dimensions of
        the video frame.
    """
   
   #window groeße im GUI
    scfx = 700 / framewidth
    scfy = 400/ frameheight

    return scfx, scfy
