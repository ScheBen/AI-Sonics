import json
import os
import numpy as np
import matplotlib.pyplot as plt
from Middlepart.AtmoApi import config
from Middlepart.AtmoApi import objects_config
from Middlepart.AtmoApi import object_tags



class AtmoProtocol():

    def __init__(self, api):
        self.api = api
        self.average_heights = {obj["name"]: obj["height_meters"] for obj in objects_config["objects"]}
        self.obj_tag_selection = {obj["name"]: obj["tags"] for obj in object_tags["objects"]}
    
    def calc_intersection(self, a: float, n: np.array) -> np.array:
        '''
        Calculates intersection between new Image plane E - which is orthogonal to x and y axis and has distance a to origin (aka viewer position) - and line G - defined by normalized direction vector n.

        Args: 
        a: Distance between viewer and image plane.
        n: Normalized direction vector.

        Returns:
        s: 3D - intersection point.
        '''

        # Ansatz Schnittpunkt Gerade mit Ebene
        # Ebenengleichung E: x3 = a
        # Geradengleichung G: np.array([r * n1, r * n2, r * n3]) = [x1, x2, x3]
        # r = a / n3

        r = a / n[2]

        s = np.array([r * n[0], r * n[1], r * n[2]])
        return s


    def create_movement_vector(self, object_protocol, screensize, screen_w, screen_h):
        '''Scales vector in dict so it points on image plane at given distance a.
        Args: 
            object_protocol: Protocol containing positions for one object.
            screensize: Wanted size of screen in meters.
            screen_w: Total width of the video in px.
            screen_h: Total height of the video in px.

        Returns: 
             positions: Vector containing scaled and reshaped object positions. Shape: [[xj], [yj], [zj], [tj]]
        '''

        # Calculate distance of image plane assuming an stereo-like distance to screen (stereo-triangle)
        a = np.tan(np.deg2rad(60)) * screensize / 2 
        
        positions = np.array([]) # Return array
        
        # # Get height of object to later estimate its depth 
        object_height_meters = self.average_heights.get(object_protocol["object class"], -1) 
        
        # # In case not depth can be calculated throw back an error
        #if object_height_meters == -1:
        #    raise ValueError("No estimated depth can be calculated for the object ", object_protocol["object class"])
        
        # Iterate through object positions
        for j in range(len(object_protocol["t"])):
            
            # Build 3d vector (stereo-triangle)
            v = np.array([object_protocol["x"][j] - screen_w / 2 , 
                        object_protocol["y"][j] + screen_h / 2 , 
                        np.tan(np.deg2rad(60)) * screen_w / 2 ]) 
            n = v / np.linalg.norm(v) # normalize v
            
            # Calculate the estimated depth           
            depth = self.depth_estimation(screen_h, object_protocol["h"][j], object_height_meters)
            
            scale = np.linalg.norm(self.calc_intersection(a, n)) + depth # Calculate scale factor depending on plane distance a and estimated depth
            vector = np.append(n * scale, object_protocol["t"][j]) # Scale vector and append timestamp j
            
            if j == 0:
                positions = np.append(positions, vector) # Add first position
            else:
                positions = np.c_[positions, vector] # Append position to vector and reshape fitting audio functions
        
        return positions


    def depth_estimation(self,image_height, height_pixels_obj, height_meters_obj, height_pixels_ref=None, height_meters_ref=1.524, depth_meters_ref=3.0):
        '''Depth estimation technique requires the comparison of two objects, one of which is a reference object, as a baseline for calculating the approximate depths of all other detected objects.
        
        Args: 
            image_height: Total height of the video in p
            height_pixels_obj: Pixel height of the bounding box of one detected object at a given time t
            height_meters_obj: Estimated height of the detected object in meters (e.g. the average height of a "person" is 1.61544 meters and that of a "car" is 1.524 meters)
            height_pixels_ref: height of the predetermined reference object in pixels
            height_meters_ref: height of the predetermined reference object in metres
            depth_metres_ref: empirically proven depth of the reference object in metres
        
        Returns: 
             depth_estimation: Vector containing the depth of the object accordning to but not including a given time t. Shape: [dj]
             
        Note: By default the reference object is of the class "car". The pixelheight of the bounding box is half of the image height and thus leads to an empirically proven actual height of 3m.
        
        '''
        
        if height_pixels_ref is None:
            height_pixels_ref = image_height / 2
        
        # Using similar triangles, the following relationship n_obj / n_ref = r_obj / r_ref is determined where n_obj is the only unknown variable that is solved for
        # n_obj and n_ref: normalised pixel height of the object or the reference
        # r_obj and r_ref: real height in meters of the object or the reference
        
        n_ref = height_pixels_ref / image_height
        
        n_obj = (height_meters_obj / height_meters_ref) * n_ref
        
        # Comparrison of the expected normalised height (n_obj) with the actual normalised height (n_obj′) of the object in the image whose depth should be estimated 
        n_obj_prime = height_pixels_obj / image_height
        factor = n_obj / n_obj_prime
        
        # Using similar triangles again, we obtain the estimated depth d_obj of object as follows d_obj = (n_obj / n_obj')*d_ref
        depth_estimation = factor * depth_meters_ref 

        return depth_estimation


    def edit_protocol(self, protocol):
        
        new_protocol = {}
        
        for i in protocol:
            print(i)
            try:
                # Detect border collision and adjust protocol accordingly
                
                protocol[i] = self.detect_border_collisions(protocol[i],config["frame_width"],config["frame_height"])
                
                vector = self.create_movement_vector(protocol[i], config["default_display_size"], config["frame_width"], config["frame_height"])#self.api.default_display_size, self.api.video_height, self.api.video_width)
                
                vector, entry_leave = self.final_vector_adjustments(vector)
                
                tags = self.create_tags(protocol[i]["object class"],vector)
                
                new_protocol[str(i)] = {"object class": protocol[i]["object class"], "vector": vector, "tags": tags, "entry_leave": entry_leave}

            except: 
                # object is not in the objectsconfig dictionary, so no new protocol is written for it.
                continue
                
        return new_protocol
    
   
    def create_tags(self, obj_class, vector):

        '''
        Divides the appearance of an object into different periods of time, 
        depending on their movement and assigns appropriate tags to the sections. 
        The tags are chosen based on an object tag json file and its conditions
        
        Args: 
            obj_class: name of the current object
            vector: position and time data of the current object
        
        Returns: 
             tags: Array with multiple tag dictionaries and their start/end timestamp indices
             
        Note: 

        '''
        #   Tag possibilities of the actual object
        if obj_class not in self.obj_tag_selection:
            return [{"name": "","start_timestamp_idx": 0,"end_timestamp_idx": -1}]

        obj_tags = self.obj_tag_selection.get(obj_class)

        vel = []
        tags = np.array([])
        foo = np.array([])
        cal_shift = np.full((1,3),15)
        ts_start = 0
        current_tag = ""
        prev_tag = ""
        changes = 0
        vel_threshold = config["velocity_threshold"]
        #print(obj_class)
        for i in range(len(vector[0])-1):
            
            #   Velocity vector calculation

            vec_1 = [vector[0][i], vector[1][i], vector[2][i]] + cal_shift
            vec_2 = [vector[0][i+1], vector[1][i+1], vector[2][i+1]] + cal_shift
            vel_vec = (vec_2 - vec_1)/(vector[3][i+1]-vector[3][i])

            #   Determination of the current velocity value via the length of the velocity vector

            current_velocity = np.linalg.norm(vel_vec)
            
            #   Tag category assignment, when there is a change in velocity based on the velocity threshold

            if current_velocity > vel_threshold and current_tag != "moving":
                prev_tag = current_tag
                current_tag = "moving"
                changes += 1
            elif current_velocity <= vel_threshold and current_tag != "stand":
                prev_tag = current_tag
                current_tag = "stand"
                changes += 1
            
            #   Add last tag to the array, if there was a tag change

            if changes > 1:
                tags = np.append(tags,{"name": prev_tag,"start_timestamp_idx": ts_start,"end_timestamp_idx": i})
                ts_start = i
                vel.append(foo)
                foo = np.array([])
                changes = 1
                
            foo = np.append(foo,current_velocity)

        vel.append(foo)
        changes = len(vel)
        tags = np.append(tags,{"name": current_tag,"start_timestamp_idx": ts_start,"end_timestamp_idx": len(vector[3])-1})
        
        #   Check the current object and its tag categories against each tag condition to select the appropriate tag

        for i in range(len(tags)):
            for obj_tag in obj_tags:
                current_tag = tags[i]["name"]
                velocity = vel[i][int(len(vel[i])/4)]
                if eval(obj_tag["condition"]):
                    tags[i]["name"] = obj_tag["value"]
                    break

        return tags

    def check_for_sequence(self, index_list, new_index):
        '''Checks last index in index_list and compares it to new_index. If new_index is one higher than the last index in index_list, it is appended to the last list in index_list. 
        If not, a new list is appended to index_list with new_index as its first element.
        Doing so, indices that are in sequence are grouped together in one list.
        Parameters:
            index_list (list): list of lists of indices
            new_index (int): index to be checked
        Returns:
            index_list (list): updated list of lists of indices
        '''
        try:
            if (index_list[-1][-1] == (new_index - 1)):
                index_list[-1].append(new_index)
            else:
                index_list.append([new_index])
        except:
            index_list.append([new_index])

        return index_list
    
    def detect_border_collisions(self, object_protocol, image_width = 854, image_height = 480):
        '''Detects collision between object bounding box and image borders and handles them.
        Parameters:
            object_protocol (dict): dictionary containing object information
            image_width (int): width of image
            image_height (int): height of image
        Returns:
            object_protocol (dict): updated dictionary containing object information
        '''
        # Check horizontal image borders
        w = object_protocol["w"].copy()
        x = object_protocol["x"].copy()
        h = object_protocol["h"].copy()
        y = object_protocol["y"].copy()

        l = x - np.array(w) / 2 # x coordinates of left border of bounding box
        r = x + np.array(w) / 2 # x coordinates of right border of bounding box
        b = y - np.array(h) / 2 # y coordinates of bottom border of bounding box
        t = y + np.array(h) / 2 # y coordinates of top border of bounding box

        left_border_collision_points = []   # List of indices of left border collisions
        right_border_collision_points = []  # List of indices of right border collisions
        top_border_collision_points = []    # List of indices of top border collisions
        bottom_border_collision_points = [] # List of indices of bottom border collisions

        threshold_width = 5 # Threshold for comparing bounding box size frame-wise
        threshold_border = 1 # Threshold for comparing bbox border to image border

        # Check for collisions with various borders
        for i in range(len(x)):
            # Left border
            try:
                if (l[i] <= threshold_border and (abs(w[i+1] - w[i]) >= threshold_width  or abs(w[i-1] - w[i]) >= threshold_width) ):
                    left_border_collision_points = self.check_for_sequence(left_border_collision_points, i)
            except:
                try:
                    if l[i] <= threshold_border and (abs(w[i+1] - w[i]) >= threshold_width):
                        left_border_collision_points = self.check_for_sequence(left_border_collision_points, i)
                except:
                    try: 
                        if l[i] <= threshold_border and (abs(w[i-1] - w[i]) >= threshold_width):
                            left_border_collision_points = self.check_for_sequence(left_border_collision_points, i)
                    except:
                        pass
            
            # Right border
            try:
                if (r[i] >= (image_width  - threshold_border) and (abs(w[i+1] - w[i]) >= threshold_width  or abs(w[i-1] - w[i]) >= threshold_width) ):
                    right_border_collision_points = self.check_for_sequence(right_border_collision_points, i)
            except:
                try:
                    if r[i] >= (image_width  - threshold_border) and (abs(w[i+1] - w[i]) >= threshold_width):
                        right_border_collision_points = self.check_for_sequence(right_border_collision_points, i)
                except:
                    try: 
                        if r[i] >= (image_width  - threshold_border) and (abs(w[i-1] - w[i]) >= threshold_width):
                            right_border_collision_points = self.check_for_sequence(right_border_collision_points, i)
                    except:
                        pass
            
            # Bottom border
            try:
                if (b[i] >= (- image_height + threshold_border) and (abs(h[i+1] - h[i]) >= threshold_width or abs(h[i-1] - h[i]) >= threshold_width)):
                    bottom_border_collision_points = self.check_for_sequence(bottom_border_collision_points, i)
            except:
                try:
                    if b[i] >= (- image_height + threshold_border) and (abs(h[i+1] - h[i]) >= threshold_width):
                        bottom_border_collision_points = self.check_for_sequence(bottom_border_collision_points, i)
                except:
                    try: 
                        if b[i] >= (- image_height + threshold_border) and (abs(h[i-1] - h[i]) >= threshold_width):
                            bottom_border_collision_points = self.check_for_sequence(bottom_border_collision_points, i)
                    except:
                        pass

            # Top border
            try:
                if (t[i] >= (- threshold_border) and (abs(h[i+1] - h[i]) >= threshold_width or abs(h[i-1] - h[i]) >= threshold_width)):
                    top_border_collision_points = self.check_for_sequence(top_border_collision_points, i)
            except:
                try:
                    if t[i] >= (- threshold_border) and (abs(h[i+1] - h[i]) >= threshold_width):
                        top_border_collision_points = self.check_for_sequence(top_border_collision_points, i)
                except:
                    try: 
                        if t[i] >= (- threshold_border) and (abs(h[i-1] - h[i]) >= threshold_width):
                            top_border_collision_points = self.check_for_sequence(top_border_collision_points, i)
                    except:
                        pass

        # Handle collisions
        # Left border

        for sequence in left_border_collision_points:
            w_ref = 0
            # try:
            #     w_ref = (w[sequence[0] - 1] + w[sequence[-1] + 1]) / 2 # Does not work if some frames are not recognized as collision
            #     print(w_ref)
            # except:
            try:
                w_ref = w[sequence[- 1] + 1]
            except:
                w_ref = w[sequence[0] - 1]
            
            #print(w_ref)

            for i in sequence:
                w_diff = (w_ref - object_protocol["w"][i]) / 2
                object_protocol["x"][i] -= w_diff
                #print(w_diff)
            
        # Right border
        for sequence in right_border_collision_points:
            w_ref = 0
            try:
                w_ref = w[sequence[-1] + 1]
            except:
                w_ref = w[sequence[0] - 1]
            
            for i in sequence:
                w_diff = (w_ref - object_protocol["w"][i]) / 2
                object_protocol["x"][i] += w_diff

        # Bottom border
        for sequence in bottom_border_collision_points:
            h_ref = 0
            try:
                h_ref = h[sequence[-1] + 1]
            except:
                h_ref = h[sequence[0] - 1]
            
            for i in sequence:
                h_diff = (h_ref - object_protocol["h"][i]) / 2
                object_protocol["y"][i] -= h_diff

        # Top border
        for sequence in top_border_collision_points:
            h_ref = 0
            try:
                h_ref = h[sequence[-1] + 1]
            except:
                h_ref = h[sequence[0] - 1]
            
            for i in sequence:
                h_diff = (h_ref - object_protocol["h"][i]) / 2
                object_protocol["y"][i] += h_diff



        return object_protocol #, left_border_collision_points #, right_border_collision_points, top_border_collision_points, bottom_border_collision_points
    
    def reject_outliers(self, data, m=2):
        '''
        Rejects outliers from data based on standard deviation and mean.
        
        Args:
            data: Numpy-Array of data to be filtered. Shape: (n, 4), where n is the number of samples and 4 is the number of features (x, y, z, t).
            m: Number of standard deviations to be used as threshold. Default: 2.
        
        Returns:
            Numpy-Array of filtered data. Shape: (n, 4), where n is the number of samples and 4 is the number of features (x, y, z, t).
        '''
        rtn_array = np.empty((1, 4))

        # Iterates over data and checks for every point, if x, y, or z is an outlier. If it is, the point is not added to the return array.
        for i in range(len(data)):
            if abs(data[i, 0] - np.mean(data[:, 0])) > m * np.std(data[:, 0]):
                #print("Hello1: ", data[i])
                pass
            elif abs(data[i, 1] - np.mean(data[:, 1])) > m * np.std(data[:, 1]):
                #print("Hello2: ", data[i])
                pass
            elif abs(data[i, 2] - np.mean(data[:, 2])) > m * np.std(data[:, 2]):
                #print("Hello3: ", data[i])
                pass
            else:
                rtn_array = np.append(rtn_array, data[i].reshape(1, 4), axis=0)
        rtn_array = np.delete(rtn_array, 0, axis = 0) # delete first row, created by np.empty
        return rtn_array

    # Polynomial fit function, returns x, y, z for t input and p parameters
    def smooth_values(self, p, sample, degree):
        '''
        Calculates smoothed values for a given sample and polynomial parameters.
        
        Args:
            p: Numpy-Array of polynomial parameters. Shape: (n, 3), where n is the degree of the polynomial.
            sample: t value, to calculate smoothed (x, y, z) for.
            degree: Degree of the polynomial.

        Returns:
            Numpy-Array of smoothed (x, y, z) values. Shape: (1, 3).
        '''

        x, y, z = 0, 0, 0
        for i in range(degree + 1):
            x += p[i, 0] * sample**(degree - i)
            y += p[i, 1] * sample**(degree - i)
            z += p[i, 2] * sample**(degree - i)
        return np.array([x, y, z])

    def final_vector_adjustments(self, vector, degree = 4, factor = 0.2, extrapolation_degree = 1):
        '''
        Takes object vector, removes outliers and calculates smoothed values, based on polynomial fit.
        Then checks, whether object comes into frame or leaves it during video. If so, extrapolates values for offscreen time values.
        
        Args:
            vector: Numpy-Array vector of the object containing position ([x], [y], [z], [t]).
            degree: Degree of the polynomial. Default: 5.
            factor: Percentage of t values that are used for extrapolation. Default: 0.1.
            extrapolation_degree: Degree of the polynomial used for extrapolation. Default: 1 (linear extrapolation).
                
        Returns:
            Numpy-Array of smoothed (x, y, z, t) values. Shape: (4, n), where n is the number of samples.
        '''
        # Remove outliers
        t0 = vector[3][0]
        t_last = vector[3][-1]
        vector = self.reject_outliers(vector.T).T
        
        t = vector[3]
        y = np.array([vector[0], vector[1], vector[2]]).T # Target values, i. e. (x, y, z) values
        p = np.polyfit(t, y, degree) # p stores polynomial fit parameters
        
        smoothed_vec = np.empty((1, 3))
        # Predict smoothed values for t values using polynomial fit parameters
        for sample in t:
            smoothed_vec = np.append(smoothed_vec, np.array([self.smooth_values(p, sample, degree)]), axis=0)
        smoothed_vec = np.delete(smoothed_vec, 0, axis = 0) # delete first row, created by np.empty
        
        ### Predict offscreen values (Pre and Post entry) using different polynomial model
        bool_array = [False, False]
        # Check if object comes into frame during video
        t_object_pre_post_notice = config["vector_manipulation_thresholds"]["t_object_pre_post_notice"] # Time in seconds object will be audible before and after it enters/leaves the frame
        percentage = int(np.floor(t.size * factor)) # Percentage of t values that are used for extrapolation
        
        if t0 > 0:
            bool_array[0] = True
            # Add t value for pre offscreen sound
            # Smooth_Values function will predict values for offscreen t values
            first_half = t[:percentage]
            print(first_half)
            p1 = np.polyfit(first_half, y[:percentage].copy(), extrapolation_degree)
            t_pre = t0 - t_object_pre_post_notice

            if t_pre < 0:
                t_pre = 0 # Prevent negative t values (t values before video start)

            y_pre = self.smooth_values(p1, t_pre, 1)
            print(y_pre)
            t = np.insert(t, 0, t_pre)
            smoothed_vec = np.insert(smoothed_vec, 0, y_pre, axis = 0)
        
        # Check if object leaves frame during video
        video_duration = config["video_duration"]
        if t_last < video_duration:
            bool_array[1] = True
            # Add t value for post offscreen sound
            # Smooth_Values function will predict values for offscreen t values
            scnd_half = t[-percentage:].copy()
            p2 = np.polyfit(scnd_half, y[-percentage:].copy(), 1)
            t_post = t_last + t_object_pre_post_notice

            if t_post > video_duration:
                t_post = video_duration # Prevent t values after video ends

            y_post = self.smooth_values(p2, t_post, 1)
            print(y_post)
            t = np.append(t, t_post)
            smoothed_vec = np.insert(smoothed_vec, smoothed_vec.shape[0], y_post, axis = 0)   

        return np.append(smoothed_vec.T, np.reshape(t, (1, t.size)), axis=0), bool_array

    def process_video_frame(self, protocol, w_video, h_video, threshold=5):
        '''
        Exrapolate the movement_vector outside of the videobountries
        
        Args:
            vector of an object in the video with the shape: [[xj], [yj], [hj], [tj]] (x, y and h are pixel values at the given time t in seconds of the video. h indicates the height of the bounding box)
            height h_video of the input-video of the objectdetection ai
            width w_video of the input-video of the objectdetection ai
            tesholdvalue in pixels sets object detection's closeness to video edge. Smaller value is precise, larger allows more distance.

        Returns: 
             Extraploated movement_vector
             
        Notes:
            1. The function calls create_movement_vector for a movement vector [[x_new], [y_new], [z], [t]] from given x, y, h, and t.
            2. Continuously checking x and y against image edges: x ± threshold = 0, y ± threshold = 0, x ± threshold = w_video, y ± threshold = h_video.
            3. If edge case, use t to extrapolate x, y, h trend from the past second to maintain trend outside image.
            4. As t progresses, x, y, h direction/change remains constant outside image.
            5. Extrapolation stops if create_movement_vector z-value > 15.
            
            Problem: What happens if object retruns into the image? 
        '''
        x, y, h, t = protocol["x"], protocol["y"], protocol["h"], protocol["t"]
       
        if abs(x) <= threshold or abs(y) <= threshold or abs(x - w_video) <= threshold or abs(y - h_video) <= threshold:
            last_positions = ([x], [y], [h])
            
            while self.create_movement_vector(protocol, (w_video, h_video), w_video, h_video)[-2] <= 15:
                x_new, y_new, h_new = self.extrapolate_values(last_positions, t, protocol["t"])
                last_positions[0].append(x_new)
                last_positions[1].append(y_new)
                last_positions[2].append(h_new)
                t += (protocol["t"] - t)  # Advance time based on the difference in t values
            return last_positions
        else:
            movement_vector = self.create_movement_vector(protocol, (w_video, h_video), w_video, h_video)
            return movement_vector

    def extrapolate_values(self, last_positions, t, t_boundary):
        # Extrapolation of x-, y-, and h-values based on the last positions
        x_values, y_values, h_values = last_positions
        delta_t = t - t_boundary
        x_extrapolated = x_values[-1] + delta_t * (x_values[-1] - x_values[-2])
        y_extrapolated = y_values[-1] + delta_t * (y_values[-1] - y_values[-2])
        h_extrapolated = h_values[-1]
        return x_extrapolated, y_extrapolated, h_extrapolated    
    

    def output_protocol(self):
        #   Für das Output Protokoll könnte man das ursprüngliche Protokoll extra abspeichern und wenn das output protokoll
        #   aufgerufen wird, wird damit ein schlichteres protokoll nur mit den tagnamen und die start + end timestamps 
        #   abgespeichert. Oder man erstellt einfach direkt das output Protokoll in einem extra dict, wenn das revised 
        #   protokoll erstellt wird. 
        
        
        return
    

    def correction_object_assignemnt(self):


        return
