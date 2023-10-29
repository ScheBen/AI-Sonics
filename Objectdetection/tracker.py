#Trackers that enable the assignment of a unique ID in object detection

import math
import numpy as np
import cv2
global objects_bbs_ids

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}      
        self.id_count = 0
        self.frameskip = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
      
        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2, ClsID = rect
            cx = (x1 + x1 + x2) // 2
            cy = (y1 + y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])                
                
                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x1, y1, x2, y2, id, ClsID])
                    same_object_detected = True
                    break

            # New object is detected,  Object gets ID
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count, ClsID])
                self.id_count += 1
                
        
        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        print(self.frameskip, "frameskip")    
        
        if self.frameskip == 0 or self.frameskip == 4:
            print("das wird ausgeführt ")
            # Update dictionary with IDs not used removed
            self.center_points = new_center_points.copy()
            self.frameskip = 0
        
        self.frameskip = self.frameskip + 1
        
        return objects_bbs_ids

class OpticalFlowTracker:
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None
    
    def track_optical_flow(self, curr_frame, prev_bbox):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_gray, self.prev_points, None)
        
        valid_curr_points = curr_points[status == 1]
        prev_bbox_center = np.mean(self.prev_points, axis=0)
        curr_bbox_center = np.mean(valid_curr_points, axis=0)
        
        distance = np.linalg.norm(curr_bbox_center - prev_bbox_center)
        
        # Update the previous frame and points for the next iteration
        self.prev_frame = curr_gray
        self.prev_points = valid_curr_points.reshape(-1, 1, 2)
        
        return distance

class EuclideanDistTrackerWithOpticalFlow:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.frameskip = 0
        self.optical_flow_tracker = OpticalFlowTracker()
    
    def update_with_optical_flow(self, objects_rect):
        objects_bbs_ids = []
        
        for rect in objects_rect:
            x1, y1, x2, y2, ClsID = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object_detected = False
            for obj_id, (prev_cx, prev_cy) in self.center_points.items():
                distance = self.optical_flow_tracker.track_optical_flow(curr_frame, [prev_cx, prev_cy, cx, cy])
                if distance < 50:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id, ClsID])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count, ClsID])
                self.id_count += 1
        
        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        if self.frameskip == 0 or self.frameskip == 4:
            # Update dictionary with IDs not used removed
            self.center_points = new_center_points.copy()
            self.frameskip = 0
        
        self.frameskip += 1
        
        return objects_bbs_ids

