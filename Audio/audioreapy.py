import reapy as rp                          # reapy wrapper for more pythonic interaction with REAPER
from reapy import reascript_api as RPR      # ReaScript functions, that are not included with reapy yet, are available in this sub-module

import numpy as np
import librosa
import soundfile as sf
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter



def create_ambisonics_master(project: rp.Project, tr_name: str) -> rp.Track:
    """Adds an Ambisonics bus to the current REAPER project. 
    Ambisonics bus will be used as master bus, actual master track will be muted
    Returns the created track object.

    Args:
        project: current REAPER project.
        tr_name: Name string the track should get.

    Returns:
        rp.Track: New created Ambisonics bus.
    """
    with rp.inside_reaper():

        # Initializing actual master track as ambisonics bus is not possible, because changing number of channels to 16 does not work
        project.master_track.set_info_value("B_MUTE", True) # Muting master track, so ambisonics playback is not disturbed
        tr = project.add_track(0, tr_name)

        tr.set_info_value("I_NCHAN", 16) # 16 Channels for 3rd order ambisonics
        

        tr.add_fx("BinauralDecoder (IEM) (64ch)") # TODO: change to other decoder, depending on playback setting
        tr.add_send() # adds hardware send

    return tr

# Adding Ambisonics track to current project

def create_ambisonics_track(project: rp.Project, tr_name: str, ambisonics_bus: rp.Track = None, n_channels: int = 16, parent_group = None) -> rp.Track: # maybe change arg tr_name to tr_index later
    """Adds an Ambisonics track to the current REAPER project.
    Adds an track, changes the number of channels, adds IEM RoomEncoder to track and initially corrects listener position.
    Returns the created track object.

    Args:
        project: current REAPER project.
        tr_name: Name string the track should get.
        n_channels: Number of channels. Default = 16 for 3rd order ambisonics
        ambisonics_bus: Track, the new created track should send to.

    Returns:
        rp.Track: New created Ambisonics Track.
    """

    # Changes the number of track channels for first track
    with rp.inside_reaper():
        tr = project.add_track(0, tr_name)
        tr.set_info_value("I_NCHAN", n_channels) # 16 Channels for 3rd order ambisonics
        tr.set_info_value("B_MAINSEND", 0) # Mute Master Send 
        fx_re = tr.add_fx("RoomEncoder (IEM) (64ch)") # Adds FX RoomEncoder to track
        # Setting listener position xyz to 0 m , number of reflections to 1
        for i in range(10, 13):
            fx_re.params[i] = 0.5 # params describes value within range from 0 to 1
        for i in range(4, 7):
            fx_re.params[i] = 1 # Setting Room Dimensions max values (30, 30, 20)
        fx_re.params[13] = 0.   # Set number of reflections to 1

        if ambisonics_bus != None:
            
            tr.add_send(ambisonics_bus)
            
            #tr.make_only_selected_track()
            #tr_RPR = RPR.GetLastTouchedTrack()

            if (n_channels == 16): # TODO: Change so send and destination channels are set correctly for other n_channels
                #RPR.SetTrackSendInfo_Value(tr_RPR, 0, 0, "I_SRCCHAN", 8192) # set send source channels to 1-16
                tr.sends[0].set_info("I_SRCCHAN", 8192)
                #RPR.SetTrackSendInfo_Value(tr_RPR, 0, 0, "I_DSTCHAN", 0) # set send destination channels to 1-16
                tr.sends[0].set_info("I_DSTCHAN", 0)

    if parent_group == None:
        return tr
    
    # Add code for organizing track groups / track_parents in REAPER project
    return tr


# Insert Media File

def insert_wav(file_path: str, pos: float,  tr: rp.Track) -> rp.Item:
    """Inserts a media file at a given position on the selected track.
    Returns the created media item.

    Args:
        file_path: Path to the media file to be inserted. Please use raw string by putting r in front of the string (r"..").
        pos: Position in seconds at which media file should be inserted.
        tr: Track on which media file should be inserted.

    Returns:
        rp.Item: Created Media Item.
    """
    with rp.inside_reaper():
        project = rp.Project()
        tr.make_only_selected_track()
        project.cursor_position = pos
        RPR.InsertMedia(file_path, 0) # Inserts file to current track (mode=0)
        item = RPR.GetSelectedMediaItem(0, 0) # Takes current project, index of selected item
        RPR.SetMediaItemSelected(item, False) # Unselect MediaItem to prevent further issues
    return item


# #Order of RoomEncoder params
# with rp.inside_reaper():
#     for i in range (36):#(track.fxs[0].n_params):
#         print(i, track.fxs[0].params[i].name)


# Automation for RoomEncoder
# Write automation for RoomEncoder (IEM) (64ch)

def write_automation(track: rp.Track, vec: np.array, fx_index: int = 0 ):
    """Creates automation for IEM RoomEncoder instance of given track, depending on given movement vector.

    Args:
        track: Ambisonics track, automation should take place on.
        vec: Vector of style [x, y, z, t] defining movement and corresponding automation. 
        fx_index: Index of the IEM RoomEncoder plugin, aka position in fx panel. Usually 0.

    """
    with rp.inside_reaper():
        track.make_only_selected_track() # Workaround, as simply handing over tr does not work
        tr = RPR.GetLastTouchedTrack()

        # Source position x
        x_env = RPR.GetFXEnvelope(tr, fx_index, 7, True)
        t = 0
        for i in vec[3]:
            RPR.InsertEnvelopePoint(x_env, float(i), float(vec[0, t]), 5, 0, False, False) # TODO: discuss wether points should be linear or have bezier tension?
            t += 1
        
        # Source position y
        y_env = RPR.GetFXEnvelope(tr, fx_index, 8, True)
        t = 0
        for i in vec[3]:
            RPR.InsertEnvelopePoint(y_env, float(i), float(vec[1, t]), 5, 0, False, False) # TODO: discuss wether points should be linear or have bezier tension?
            t += 1
        
        # Source position z
        z_env = RPR.GetFXEnvelope(tr, fx_index, 9, True)
        t = 0 
        for i in vec[3]:
            RPR.InsertEnvelopePoint(z_env, float(i), float(vec[2, t]), 5, 0, False, False) # TODO: discuss wether points should be linear or have bezier tension?
            t += 1
            
    return

# Volume Automation
def add_fade(p1, p2, track_name):
    ''' Writes volume automation for two points on given track. Volume will fade in/out from on point to the other.
    Args:
        p1: This point is set to -inf dB. Shape: (t)
        p2: This point is set to 0 dB. Shape: (t)
        track_name: Name of the track the volume automation will be written to.
    '''

    with rp.inside_reaper():
        project = rp.Project()
        track = project.tracks[track_name]
        track.make_only_selected_track()
        tr = RPR.GetLastTouchedTrack()
        if (len(track.envelopes) == 0):    
            RPR.Main_OnCommand(40406,0) # Toggle volume envelope
        elif (track.envelopes[0].name != "Volume"):
            RPR.Main_OnCommand(40406,0) # Toggle volume envelope

        vol_env = RPR.GetTrackEnvelopeByName(tr, "Volume")
        if (p1 < p2): # Different bezier tension depending on fade direction
            RPR.InsertEnvelopePoint(vol_env, p1, 0, 5, -0.5, False, False)
            RPR.InsertEnvelopePoint(vol_env, p2, 716.3, 5, 0, False, False)
        else:
            RPR.InsertEnvelopePoint(vol_env, p1, 0, 5, 0, False, False)
            RPR.InsertEnvelopePoint(vol_env, p2, 716.3, 5,  0.5 , False, False)

  
    return

# Change Fader Value

def fader_volume(track: rp.Track, vol: float):
    """Changes the fader volume of given track to given volume value.

    Args:
        track: Track whose fader is to be changed.
        vol: Volume value to be set. Value logic: 1 = 0dB, 0.5 = -6dB, 2 = +6dB etc.

    """
    with rp.inside_reaper():
        # vol = 1. # Float value: 1 = 0.dB, 0.5 = -6dB , 2 = +6dB etc. 
        track.set_info_value("D_VOL", vol)
    return


## Onset Detection

def cut_to_onset(sample_path: str) -> str:
    """Detects onset in given audio file, trims start to first onset and saves a new file.
    Args: 
        sample_path: File path to the media file to be cut to onset.

    Returns: str: Path to the edited audio file.
    """
    sr = librosa.get_samplerate(path=sample_path)
    y, sr = librosa.load(path=sample_path, sr=sr) # load file for following onset detection

    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_raw = librosa.onset.onset_detect(onset_envelope=o_env, backtrack=False, units='frames')
    onset_bt = librosa.onset.onset_backtrack(onset_raw, o_env)

    ytrimmed = y[librosa.frames_to_samples(onset_bt)[0]:] # Trim y start to first backtracked onset detected
    
    # Create new filename
    file_name = os.path.basename(sample_path)
    splits = os.path.splitext(file_name)
    new_file_name = splits[0]+"_trimmed"+splits[1]
    sf.write(new_file_name, ytrimmed, sr)

    return os.getcwd()+ '\\' + new_file_name
    

## Scene Sound 

def create_scene_audio_ME(sample_paths: list[str, str, str], video_duration: float, 
                       bus: rp.Track, tr_name: str = "scene", video_start: float = 0, fade_duration: float = 3, cut_to_onset: bool = False) -> rp.Track:
   
    '''Creates 3 children stereo tracks that send to ambisonics parent track. Stereo samples are then arranged around the listener, using IEM MultiEncoder.
    
        Args:
        sample_paths: List containing paths to 3 stereo samples.
        video_duration: Duration in seconds the audio track should cover.
        bus: Ambisonics bus the parent track will send to.
        
            Optional: 
            tr_name: Name of the parent track. Default is "scene".
            video_start: Timestamp in seconds audio should start. Default = 0 .
            fade_duration: Duration in seconds of crossfade that is applied, when audio sample is shorter than video_duration. Default = 3 .
            cut_to_onset: Define whether audio samples should be cut_to_onset. Not necessary if samples are already trimmed. Takes additional computation time and disk space. Default = True .

        Returns:
        rp.Track: Parent track.
    '''
    with rp.inside_reaper():
        project = rp.Project()
        # Create 3 stereo tracks and 1 16 channel ambisonics track. 
        child3 = create_ambisonics_track(project, "sample3", n_channels=2)
        child2 = create_ambisonics_track(project, "sample2", n_channels=2)
        child1 = create_ambisonics_track(project, "sample1", n_channels=2)

        child_tracks = [child1, child2, child3]

        for path, track in zip(sample_paths, child_tracks):
            if (cut_to_onset == True):
                path = cut_to_onset(path)
            track.fxs[0].delete()
            sample_length = librosa.get_duration(path=path)
            insert_wav(path, video_start, track)
            new_pos = video_start

            shortage = sample_length - video_duration

            if (shortage < 0):
                # Times of additional samples needed to fill vector length
                n_samples = (video_duration - sample_length) / (sample_length - fade_duration)
                j = int(np.ceil(n_samples)) # Rounding up n to get needed loop iterations

                for i in range(j):
                    new_pos += sample_length - fade_duration # Calculating and iterating position of next sample using crossfade
                    insert_wav(path, new_pos, track)
                track.items[-1].length = (sample_length - fade_duration) * (n_samples - np.floor(n_samples)) + fade_duration # When all samples placed, cut last samples exceed to fit vector size
            elif (shortage == 0):
                return
            elif (shortage > 0):
                track.items[-1].length = video_duration

        parent = create_ambisonics_track(project, tr_name, ambisonics_bus = bus)

        # Group stereo tracks as children of ambisonics track
        parent.set_info_value("I_FOLDERDEPTH", 1)
        child3.set_info_value("I_FOLDERDEPTH", -1)

        # Change parent sends to distinct channels by changing parent send offset
        offset = 2
        for child in [child2, child3]:
            child.set_info_value("C_MAINSEND_OFFS", offset)
            offset += 2

        # Change send of parent track to ambisonics bus to correct channel config (1-6)
        parent.sends[0].set_info("I_SRCCHAN", 3072)
        parent.sends[0].set_info("I_DSTCHAN", 0)

        # Remove RoomEncoder and add MultiEncoder to parent track
        parent.fxs[0].delete()
        fx_re = parent.add_fx("MultiEncoder (IEM) (64ch)")

        # Set number of input channels to 6 (by setting it to 0.1)
        fx_re.params[0] = 0.1

        # Set channel azimuth to values seperated by 60 degrees
        fx_indices = range(7, 33, 5)
        azimuths = [0.58333, 0.41666, 0.25, 0.08333, 0.91666, 0.75] # equals in degrees [30, -30, -90, -150, 150, 90]
        for i, j in zip(fx_indices, azimuths):
            fx_re.params[i] = j

    return parent


def create_scene_audio(sample_path: str, video_duration: float, 
                       bus: rp.Track, tr_name: str = "scene", video_start: float = 0, fade_duration: float = 3, cut_to_onset: bool = False) -> rp.Track:
   
    '''Creates first order ambisonics B-format track that send to ambisonics parent track.
    
        Args:
        sample_path: Path to scene audio file.
        video_duration: Duration in seconds the audio track should cover.
        bus: Ambisonics bus the parent track will send to.
        
            Optional: 
            tr_name: Name of the parent track. Default is "scene".
            video_start: Timestamp in seconds audio should start. Default = 0 .
            fade_duration: Duration in seconds of crossfade that is applied, when audio sample is shorter than video_duration. Default = 3 .
            cut_to_onset: Define whether audio samples should be cut_to_onset. Not necessary if samples are already trimmed. Takes additional computation time and disk space. Default = True .

        Returns:
        rp.Track: scene audio track.
    '''
    with rp.inside_reaper():
        project = rp.Project()
        # Create 3 stereo tracks and 1 16 channel ambisonics track. 
        track = create_ambisonics_track(project, tr_name, n_channels=4, ambisonics_bus=bus)

        if (cut_to_onset == True):
            sample_path = cut_to_onset(sample_path)
        track.fxs[0].delete()
        sample_length = librosa.get_duration(path=sample_path)
        insert_wav(sample_path, video_start, track)
        new_pos = video_start

        shortage = sample_length - video_duration

        if (shortage < 0):
            # Times of additional samples needed to fill vector length
            n_samples = (video_duration - sample_length) / (sample_length - fade_duration)
            j = int(np.ceil(n_samples)) # Rounding up n to get needed loop iterations

            for i in range(j):
                new_pos += sample_length - fade_duration # Calculating and iterating position of next sample using crossfade
                insert_wav(sample_path, new_pos, track)
            track.items[-1].length = (sample_length - fade_duration) * (n_samples - np.floor(n_samples)) + fade_duration # When all samples placed, cut last samples exceed to fit vector size
        elif (shortage == 0):
            return
        elif (shortage > 0):
            track.items[-1].length = video_duration


        # Change send of track to ambisonics bus to correct channel config (1-4)
        track.sends[0].set_info("I_SRCCHAN", 2048)
        track.sends[0].set_info("I_DSTCHAN", 0)

    return track


## Object Sound

def create_object_audio(obj_vector: np.array, track_name: str, bus: rp.Track): # TODO: Funktionsnamen überdenken
    """Creates empty track, fitting in length to corresponding movement vector. Writes RoomEncoder automation based on movement vector.

    Args:
        obj_vector: Movement vector of object detected by neural network.
        track_name: Name the new track should recive.
        bus: Reaper Track object that is used as Ambisonics bus. Can be created using create_ambisonics_bus().
    """
    
    # Swap vector coordinates because RoomEncoder uses different coordinate system
    temp = obj_vector[0].copy()
    obj_vector[0] = (obj_vector[2] + 15) / 30   # Normalizing to range(0, 1), representing values in range(-15, 15)
    obj_vector[2] = (obj_vector[1] + 10) / 20   # Normalizing to range(0, 1), representing values in range(-10, 10)
    obj_vector[1] = (-temp + 15) / 30           # Inverting (bc of other axis direction), normalizing to range(0, 1), representing values in range(-15, 15)

    with rp.inside_reaper():
        project = rp.Project()
        track = create_ambisonics_track(project, track_name, bus)
        write_automation(track, obj_vector)

    return

def add_sample_to_track(sample_path: str, start_timestamp, end_timestamp, track_name: str, fade_duration: float = 3):
    """Adds media item to existing track, fitting in length to corresponding tag timestamps. This is done by looping given sample several times.
    Args: 
        sample_path: File path to the media file to be inserted. Note: use raw string by putting r in front of the string (r"..").
        start_timestamp: Timestamp in seconds the sample should start.
        end_timestamp: Timestamp in seconds the sample should end.
        track_name: Name of the track the sample should be added to.
        fade_duration: Duration in seconds of crossfade that is applied, when audio sample is shorter than video_duration. Default = 3 .
    """

    with rp.inside_reaper():
        project = rp.Project()
        track = project.tracks[track_name]
        new_pos = start_timestamp # Position to place first sample
        insert_wav(sample_path, new_pos, track)
        sample_length = librosa.get_duration(path = sample_path) # in seconds

        shortage = sample_length - end_timestamp - start_timestamp # Difference between sample length and object appearance duration: Are additional samples needed?

        if (shortage < 0 ) :
            n_samples = ((end_timestamp - start_timestamp) - sample_length) / (sample_length - fade_duration) # Times of additional samples needed to fill vector length
            j = int(np.ceil(n_samples)) # Rounding up n to get needed loop iterations
            for i in range (j) :
                new_pos += sample_length - fade_duration # Calculating and iterating position of next sample using crossfade
                insert_wav(sample_path, new_pos, track)
            track.items[-1].length = (sample_length - fade_duration) * (n_samples - np.floor(n_samples)) + fade_duration # When all samples placed, cut last samples exceed to fit vector size
        elif (shortage == 0):
            return
        elif (shortage > 0):
            track.items[-1].length = end_timestamp - start_timestamp # Cutting samples exceed to fit vector size
        #print(track.items[-1].length)
        
    return
## Plot Object Movement

def plot_animated_object_movement(object_vec):
    '''
    Produces an animated Plot of the Object position.

    Args: 
    object_vec: Numpy-Array vector of the object containing position ([x], [y], [z], [t]).

    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    def update_point(i):
        ax.clear()
        point = object_vec[0][i], object_vec[1][i], object_vec[2][i]

        ax.set(xlim3d=(0, 1), xlabel='X')
        ax.set(ylim3d=(0, 1), ylabel='Y')
        ax.set(zlim3d=(0, 1), zlabel='Z')
        points = ax.plot(point[0], point[1], point[2], label='object movement', marker = 'o')
        listener = ax.plot(0.5, 0.5, 0.5, label='listener position', marker = 'o', color='r')
        ax.legend()

        return points, listener



    #ax.plot((0.5, 0.5, 0.5), marker='o')


    ani = FuncAnimation(fig, update_point, 50, interval=100)

    ani.save("simple_animation.gif", dpi=300,
            writer=PillowWriter(fps=3))


#### Colored Scatter Plot

def plot_object_movement_scatter(object_vec):
    '''
    Produces an colored plot of the Object position. Note: Swaps y and z values internally so y is shown as vertical axis.

    Args: 
    object_vec: Numpy-Array vector of the object containing position ([x], [y], [z], [t]).

    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter(object_vec[0], object_vec[2], object_vec[1], c = object_vec[3], label = 'object position')
    ax.plot(0, 0, 0, c = 'r', marker = 'o', label = 'listener position')
    ax.set(ylabel="z", xlabel="x", zlabel="y")
    ax.legend()
    ax.dist = 11
    cbar = fig.colorbar(img)
    cbar.set_label('time', rotation = 90)


def scene_rotation_from_to(from_vec,object_vec, scene_tag):
    
    mean_x = np.mean(object_vec[2])
    mean_z = -np.mean(object_vec[0])
    mean_vec = [mean_x, 0.0, mean_z]
    norm_vec = mean_vec / np.linalg.norm(mean_vec)
    
    dot_prod = np.clip(np.dot(from_vec, norm_vec), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_prod))
    
    print(mean_vec)
    print(norm_vec)
    if mean_z < 0.0:
        angle *= -1.0
    
    project = rp.Project()    
    track = project._get_track_by_name("Scene "+scene_tag)
    fx = track.add_fx("SceneRotator (IEM) (64ch)")
    
    fx.params[2] = (angle+180)/360
