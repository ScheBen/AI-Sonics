import audiodbclient as db
import audioreapy as ar
import os
import json
import requests
import reapy as rp
import reapy.reascript_api as RPR
import numpy
from Middlepart.AtmoApi import scene_profiles
from Middlepart.AtmoApi import config
import platform


def create_project(projectPath: str, folderName, video_path: str):
    """Creates project folder and reaper project at given path. Folder is used for all needed samples and Reaper-Project file.

    Args: 
        projectPath: Parent directory / Path to new project folder.
        folderName: Name of new folder.

    Returns: 
        rp.Project(): Reaper project.
    """

    path = os.path.join(projectPath, folderName)
    if os.path.exists(path):
        count = 1
        while count > 0:
            if os.path.exists(path+str(count)):
                count += 1
            else:
                path += str(count)
                os.mkdir(path)
                folderName += str(count)
                break
    else:
        os.mkdir(path)

    # Create and save new Reaper project
    RPR.Main_OnCommand(40023,0)
    #RPR.Main_SaveProjectEx(0,path+"/"+folderName+".rpp",0)

    # Add video to Reaper project
    RPR.InsertMedia(video_path, 1) # mode = 1 -> add to new track
    if (RPR.GetToggleCommandState(50125) == 0):
        RPR.Main_OnCommand(50125,0) # Toggle video view window

    return path, folderName

def load_sample(sample_info,project_path):

    """Load sample from a sample database, based on the given uri and move the file into the current project folder.

    Args: 
        sample_info: dict with infos about the files to be loaded
        project_path: Path of the current Reaper orject

    Returns: 
        Current file location
    """
    
    #   get sample from database

    file = requests.get(sample_info['path'],allow_redirects=True, verify=False)

    #   create filename with tags and sample id

    filename = ""
    for tagnamen in sample_info['tagnamen']:
        filename += tagnamen+"_"

    filename += str(sample_info['sampleid'])+"."+sample_info['format']

    #   check if file already exist in projekt and return the path

    exist_file = os.path.isfile(os.path.join(project_path,filename))

    if exist_file:
        print(exist_file)
        return os.path.join(project_path,filename)
    
    #   save file in the reaper project folder

    open(os.path.join(project_path,filename), 'wb').write(file.content)

    return os.path.join(project_path,filename)
    

def save_project():

    RPR.Main_SaveProject(0,False)

 
def edit_project(scene_tag, objectLog, duration, project_path, video_path):
    
    """
        Edit the current Reaper project based on the video infos
    Args: 
        scene_tag: Name of the recognized scene
        objectLog: infos about the recognized objects, their position, tags and timestamps 
        duration:  Video duration
        project_path:  Reaper project path
        video_path:  Video location

    Returns: 
        
    """

    # We need reaper information about current project and create ambisonics bus to create and route object tracks
    # Maybe create in other Function?
    project = rp.Project()
    bus = ar.create_ambisonics_master(project, "Ambisonics Bus")

    # Get video filename to find the video track and set their volume to -inf

    vid_file_name = video_path[video_path.rfind('/')+1:video_path.rfind('.')]
    videotrack = project._get_track_by_name(vid_file_name)
    ar.fader_volume(videotrack,0)

    # Reaper project changes based on the current scene and their profiles.
    # Divided into pre and post changes, in order to change e.g. scene tags based on the recognised objects 
    # before searching for the scene samples or change the scene rotation after adding the scene

    scene_profile = scene_profiles[scene_tag]
    post_changes_obj = [obj["name"] for obj in scene_profile["post_changes"]]
    pre_changes_obj = [obj["name"] for obj in scene_profile["pre_changes"]]
    scene_tags = [scene_tag]
    
    # Looking for pre changes based on the profile 

    for i in objectLog:
        print(objectLog[i]["object class"])
        if objectLog[i]["object class"] in pre_changes_obj:
            obj_index = pre_changes_obj.index(objectLog[i]["object class"])
            
        # Execute the given expressions

            for sc_func in scene_profile["pre_changes"][obj_index]["expressions"]:
                eval(sc_func)
            continue

    print(scene_tags)
    # Add audio for video scene
    for i in [scene_tag]:
        
        scene_sample_arr = db.search_scene_file(scene_tags, int(duration))

        if len(scene_sample_arr) == 1:
            # Ambisonics sample exists -> add ambisonics sample to reaper
            sample_path = load_sample(scene_sample_arr[0], project_path)
            tr_name = "Scene " + scene_tag 
            ar.create_scene_audio(sample_path, duration, bus, tr_name, cut_to_onset=False)
        
        elif len(scene_sample_arr) >= 3:
            # No ambisonics sample exists -> use Multiencoder with 3 samples
            sample_paths = load_sample(scene_sample_arr, project_path) # TODO: Has to return list of paths
            tr_name = "Scene " + scene_tag 
            ar.create_scene_audio_ME(sample_paths, duration, bus, tr_name, cut_to_onset=False)


    # Add audio for objects in video
    for i in objectLog:
        #print("############## objectLog ##############\n " + str(objectLog))

        # Looking for post changes based on the profiles

        if objectLog[i]["object class"] in post_changes_obj:
            
            obj_index = post_changes_obj.index(objectLog[i]["object class"])

        # Execute the given expressions if these have not yet been carried out
        # Then it should be checked whether a sample should be searched for the object tag or not
            
            if eval(scene_profile["post_changes"][obj_index]["executed"]):
                continue

            for sc_func in scene_profile["post_changes"][obj_index]["expressions"]:
                eval(sc_func)

            scene_profile["post_changes"][obj_index]["executed"] = "True"

            if not eval(scene_profile["post_changes"][obj_index]["search_sample"]):
                continue

        print(objectLog[i]["tags"])

        entry_leave = objectLog[i]["entry_leave"] # [entry: bool, leave: bool]
        obj_vector = objectLog[i]["vector"]
        track_name = objectLog[i]["object class"] + " " + str(i)

        # Empty object track is created
        ar.create_object_audio(obj_vector, track_name, bus)

        for tag in objectLog[i]["tags"]:
            start_idx = tag["start_timestamp_idx"]
            end_idx = tag["end_timestamp_idx"]
            if(len(tag["name"]) > 0):
                obj_sample = db.search_object_file([objectLog[i]["object class"], tag["name"]], int(objectLog[i]["vector"][3][end_idx] - objectLog[i]["vector"][3][start_idx])) # Get Duration by subtracting first time stamp of last time stamp
            else:
                obj_sample = db.search_object_file([objectLog[i]["object class"]], int(objectLog[i]["vector"][3][end_idx] - objectLog[i]["vector"][3][start_idx])) # Get Duration by subtracting first time stamp of last time stamp

            # For all tags, if sample found, sample is added to object track
            if len(obj_sample) > 0:
                # Object sample exists -> add sample to object track
                #print("############## obj_sample ##############\n " + str(obj_sample))
                sample_path = load_sample(obj_sample, project_path)
                ar.add_sample_to_track(sample_path, objectLog[i]["vector"][3][start_idx], objectLog[i]["vector"][3][end_idx], track_name)
                
        # Add volume fades if object enters or leaves the scene
        if entry_leave[0] == True:
            ar.add_fade(obj_vector[3][0], obj_vector[3][1], track_name) # Takes predicted and first real t value
        if entry_leave[1] == True:
            ar.add_fade(obj_vector[3][-1], obj_vector[3][-2], track_name) # Takes predicted and last real t value
    os.path.join(config["render_strings"]["RENDER_FILE"],config["render_strings"]["RENDER_PATTERN"])
    RPR.Main_SaveProjectEx(0,config["render_strings"]["RENDER_FILE"]+"/"+config["render_strings"]["RENDER_PATTERN"]+".rpp",0)


def render_project(render_values, render_strings,duration, video_path):
    
    """
        Render reaper project based on the config parameters 
    Args: 
        render_values: values of settings based on numbers
        render_strings: values of settings based on strings

    Returns: 
        
    """
    
    project = rp.Project()
    print(project.get_info_string("RENDER_FORMAT"))
    project.time_selection = 0,duration
    project.master_track.set_info_value("I_NCHAN",render_values["RENDER_CHANNELS"])
    project.cursor_position = 0

    for key, value in render_values.items():
        project.set_info_value(key,value)

    for key, value in render_strings.items():
        project.set_info_string(key,value)

    RPR.Main_OnCommand(42230,0)


