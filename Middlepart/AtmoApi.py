import os
import sys
import ast
import json
from time import sleep
import subprocess
import matplotlib.pyplot as plt
import numpy as np
#import GUI

with open(os.path.join(os.getcwd(),"System","config.json")) as config_file:
        config = json.load(config_file)

with open(os.path.join(os.getcwd(),"System","objectsconfig.json")) as objects_config_file:
    objects_config = json.load(objects_config_file) 

with open(os.path.join(os.getcwd(),"System","objecttags.json")) as objects_tags_file:
    object_tags = json.load(objects_tags_file) 

with open(os.path.join(os.getcwd(),"System","sceneprofiles.json")) as scene_profiles_file:
    scene_profiles = json.load(scene_profiles_file) 

r_path = config['reaper_dir']

if os.path.exists(r_path) is False: 
    print("pfad existiert nicht")
    for root, dirs, files in os.walk("C:"):
        for name in files:
            if name == "reaper.exe":
                # Absolute Path reaper
                print("gefunden")
                r_path = os.path.abspath(os.path.join(root, name))
            
                break
        else:
            continue
        break

subprocess.Popen(r_path)
sleep(2)

import AtmoProtocol 
import GUI 
import Audio.audioreapy as audioreapy
import Audio.audiomixing as audiomixing
import cv2

# import YOLO_Objectdetection as obj_det
#print("###################### New Run ######################")

class AtmoApi():

    
    video_path = ""
    video_art = None
    video_width = None
    video_height = None
    
    def __init__(self,gui):
        self.a = ""
        self.atmo_protocol = AtmoProtocol.AtmoProtocol(self)
        self.gui = gui

        
    def create_atmo(self, scene, obj_protocol, video_path, duration):
        
        self.video_path = video_path
        config["video_duration"] = duration


        #   Revise object log to determine object motion vectors and distance.
        #   Position outliers are also corrected
       
        revised_obj_protocol = self.atmo_protocol.edit_protocol(obj_protocol)
        
        print(revised_obj_protocol)
        #   Start reaper and create project

        project_path, project_name = audiomixing.create_project(os.path.join(os.getcwd(),"Projects"), config['project_name'], self.video_path)
        config["render_strings"]["RENDER_FILE"] = project_path
        config["render_strings"]["RENDER_PATTERN"] = project_name
        
        print("render infos")
        print(config["render_active"])

        # #   audio mixing using the object protocol and the scene tag
        audiomixing.edit_project(scene, revised_obj_protocol, duration, project_path, video_path)

        self.output_used_settings()        

        if config["render_active"]:
            audiomixing.render_project(config["render_values"],config["render_strings"],duration, video_path)
            self.gui.process_end(project_path) 


    def output_used_settings(self):

        o_file = open(config["render_strings"]["RENDER_FILE"]+"/"+config["render_strings"]["RENDER_PATTERN"]+"_Settings.txt", "a")

        for key in config['default_settings']:
            o_file.write(key+" : "+str(config[key])+", \n")
        
        o_file.close()


if __name__ == "__main__":
    os.path.join(os.getcwd(),"System","sceneprofiles.json")

    with open(os.join(os.getcwd(),"Input","dict_Langenargen_Bodensee_bla.txt")) as strVid:
        data = strVid.read()

    obj_protocol = ast.literal_eval(data)
    scene_tag = 2
    duration = 15

    gui = GUI.VideoPlayerApp()
    atmo_api = AtmoApi(gui)
    atmo_api.config["video_duration"] = duration
    atmo_api.create_atmo(scene_tag,obj_protocol,os.join(os.getcwd(),"Input","Langenargen_Bodensee_am_Strand_Kurz_und_Knapp.mp4"),duration)
