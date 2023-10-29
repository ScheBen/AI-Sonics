import sys
import os
sys.path.append(os.path.join(os.getcwd(),"Objectdetection"))
sys.path.append(os.path.join(os.getcwd(),"Audio"))
sys.path.append(os.path.join(os.getcwd(),"Scenedetection"))
sys.path.append(os.path.join(os.getcwd(),"Middlepart"))

import Middlepart.AtmoApi as AtmoApi
from Middlepart.AtmoApi import config

import tkinter as tk
from tkinter import filedialog
import cv2
from tkVideoPlayer import TkinterVideo
import customtkinter
import threading
from PIL import Image, ImageTk
import Objectdetection.YOLO_Objectdetection  as OD
import json

from tkinter import PhotoImage
from tkinter import Canvas
from tkinter import ttk 

import Scenedetection.sceneDetection as SD
import subprocess
import platform

class VideoPlayerApp:
    def __init__(self, *args):

        if(len(args)== 0):
            return

        self.root = args[0]
        
        # Fenstergröße auf 70% des Bildschirms einstellen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = int(screen_width * 0.7)
        height = int(screen_height * 0.7)
        self.root.geometry(f"{width}x{height}")
        
        self.video_player = args[1]   
        self.root.title("AI Sound-Generator")        
        self.video_player.place(x=0, y=0, relwidth=1, relheight=1)
        self.video_player.load(os.path.join("Objectdetection","GUI_Assets","komp.mp4"))
        self.video_player.play()

        self.root.after(3000, self.show_sound_generator_app)
        
    def show_sound_generator_app(self):
        self.root.attributes('-topmost',False)
        self.video_player.stop()
        self.video_player.destroy()
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = int(screen_width * 0.7)
        height = int(screen_height * 0.7)
        
        # Hintergrundbild einstellen        
        bg_image = Image.open(os.path.join("Objectdetection","GUI_Assets","isbg.jpg"))
        bg_image = bg_image.resize((width, height))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.background_label = tk.Label(root, image=self.bg_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Defining variables
        video_path = None
        self.playing = False
        
             
        # Create a frame to center the video frame
        self.center_frame = tk.Frame(self.root)
        self.center_frame.pack(expand=True,padx=0,pady=0)      
        
        # Centering the video frame within center_frame
        self.video_frame = tk.Frame(self.center_frame, width=700, height=400)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack()
        
        chooseButton_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","my_video.png")).resize((40, 30)))
        self.choose_button = customtkinter.CTkButton(self.center_frame, text="Video auswählen", font=("Calibri Bold", 20), image=chooseButton_image, command=self.choose_video, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.choose_button.pack(pady=(20, 0)) 
        
        createSound_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","ifunkel.png")).resize((20, 20)))   
        self.play_button = customtkinter.CTkButton(self.center_frame, text="Create Sound", font=("Calibri Bold", 20), image=createSound_image, command=self.play_video, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.play_button.pack(pady=(10, 5))
        self.play_button.configure(fg_color="grey") 
        self.play_button.configure(state=tk.DISABLED)    
        
        #self.stop_button = tk.Button(center_frame, text="Stop", command=self.stop_video)
        #self.stop_button.pack(pady=10)
   
    
    
    def choose_video(self):
        global video_path
                        
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        self.show_video()
        
    
    def show_video(self): 
        
        global video_path
        if video_path:      
            self.playing = False
            print(video_path, "video_path")
            
            self.show_thumbnail()
            self.choose_button.configure(state=tk.DISABLED) 
            self.choose_button.configure(fg_color="grey") 
            self.choose_button.pack_forget()   
            self.play_button.pack_forget()
            
            self.render_frame = tk.Frame(self.center_frame)
            self.render_frame.pack(pady=(0, 10), padx=(0,0))
            
            createSound_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","sfunkeln.png")).resize((40, 40)))   
            self.play_button = customtkinter.CTkButton(self.render_frame, text="Create Sound", font=("Calibri Bold", 20), image=createSound_image, command=self.play_video, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
            self.play_button.configure(state=tk.NORMAL) 
            self.play_button.configure(fg_color="#0C1745") 
            self.play_button.configure(image=createSound_image) 
            self.play_button.pack(side=tk.LEFT, pady=(10, 0))
            
            settings_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","settings.png")).resize((30, 30)))
            self.settings_button = customtkinter.CTkButton(self.render_frame, text="", font=("Calibri Bold", 20), image=settings_image, command=self.get_settings, width=50, height=50, compound="left", hover_color="#6CC5D8", fg_color="#6CC5D8")
            self.settings_button.pack(side=tk.LEFT, pady=(10, 0), padx=(0,10))  
            
            #self.set_config_to_default()
                
    def set_config_to_default(self):

        for key in config['default_settings']:
            config[key] = config['default_settings'][key]

        if config['render_active'] is True:
            self.checkbox_variable.set(1)
        else:
            self.checkbox_variable.set(0)

        self.slider.set(config['min_confidence_threshold'])

        self.slider2.set(config['scene_accuracy'])

        self.slider3.set(config['default_display_size'])

            
    def show_video2(self):
        self.settings_button.pack_forget() 
        self.settings_label.pack_forget()  
        
        
        self.render_frame.pack_forget()  
        self.backbutton1_button.pack_forget()
        self.backbutton2_button.pack_forget() 
         
        self.video_label2.pack_forget()  
        
        
        # Centering the video frame within center_frame
        self.video_frame = tk.Frame(self.center_frame, width=700, height=400)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack()        
        
        global video_path
        
        if video_path:      
            self.playing = False
            print(video_path, "video_path")
                        
            self.show_thumbnail()
            
            self.render_frame = tk.Frame(self.center_frame)
            self.render_frame.pack(pady=(0, 10), padx=(0,0))
            
            createSound_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","sfunkeln.png")).resize((40, 40)))   
            self.play_button = customtkinter.CTkButton(self.render_frame, text="Create Sound", font=("Calibri Bold", 20), image=createSound_image, command=self.play_video, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
            self.play_button.configure(state=tk.NORMAL) 
            self.play_button.configure(fg_color="#0C1745") 
            self.play_button.configure(image=createSound_image) 
            self.play_button.pack(side=tk.LEFT, pady=(10, 0))
            
            set_image2 = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","settings.png")).resize((30, 30)))
            self.set_button2 = customtkinter.CTkButton(self.render_frame, text="", font=("Calibri Bold", 20), image=set_image2, command=self.get_settings, width=50, height=50, compound="left", hover_color="#6CC5D8", fg_color="#6CC5D8")
            self.set_button2.pack(side=tk.LEFT, pady=(10, 0), padx=(0,10))  
  
    def update_label(value):
        slider_label.config(text=f"Slider Value: {value:.2f}")    
    
    def get_settings(self):
        self.video_frame.destroy()  
        self.choose_button.destroy()
        self.play_button.destroy()
        self.settings_button.destroy()
        self.render_frame.destroy()
        
        # Label für Einstellungen
        self.settings_label = tk.Label(self.center_frame, text="Settings:", font=("Calibri Bold", 18), fg="#0C1745")
        self.settings_label.pack(pady=(10, 5))

        # Zeile mit "Rendern", Checkbox, Schieberegler und Label
        self.render_frame = tk.Frame(self.center_frame)
        self.render_frame.pack(pady=(0, 10), fill=tk.X, anchor=tk.CENTER)

        self.render_label = tk.Label(self.render_frame, text="AutoRender:", font=("Calibri Bold", 14))
        self.render_label.pack(side=tk.LEFT, padx=(0, 10))
        self.checkbox_variable = tk.IntVar()

        if config['render_active'] is True:
            self.checkbox_variable.set(1)
        else:
            self.checkbox_variable.set(0)

        self.render_checkbox = tk.Checkbutton(self.render_frame, variable=self.checkbox_variable)
        
        self.render_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        self.slider_label = tk.Label(self.render_frame, text="Min confidence:", font=("Calibri Bold", 14))
        self.slider_label.pack(side=tk.LEFT, padx=(40, 10))

        self.slider = tk.Scale(self.render_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.slider.set(config['min_confidence_threshold'])
        self.slider.pack(side=tk.LEFT) 
        
        self.slider_label2 = tk.Label(self.render_frame, text="Scene Accuracy:", font=("Calibri Bold", 14))
        self.slider_label2.pack(side=tk.LEFT, padx=(40, 10))

        self.slider2 = tk.Scale(self.render_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL)
        self.slider2.set(config['scene_accuracy'])
        self.slider2.pack(side=tk.LEFT)         

        self.slider_label3 = tk.Label(self.render_frame, text="Display Size (m):", font=("Calibri Bold", 14))
        self.slider_label3.pack(side=tk.LEFT, padx=(40, 10))

        self.slider3 = tk.Scale(self.render_frame, from_=0.1, to=3, resolution=0.01, orient=tk.HORIZONTAL)
        self.slider3.set(config['default_display_size'])
        self.slider3.pack(side=tk.LEFT)   

        back2 = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","sback.png")).resize((10, 10))) 


        self.backbutton1_button = customtkinter.CTkButton(self.center_frame, text="Default settings", font=("Calibri Bold", 20), command=self.set_config_to_default, width=100, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.backbutton1_button.pack(pady=(10, 5), padx=(10,10),anchor='s')

        self.backbutton2_button = customtkinter.CTkButton(self.center_frame, text="Save settings", font=("Calibri Bold", 20), image=back2, command=self.save_settings, width=100, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.backbutton2_button.pack(pady=(10, 5),padx=(10,10),after=self.backbutton1_button)


        
        
    def save_settings(self):       
                
        auto_render = self.checkbox_variable.get()  
        min_confidence = self.slider.get()  
        accuray_scene = self.slider2.get()  
        display_size = self.slider3.get()
        print("autorender "+str(auto_render))
        # Je nach Bedingung den JSON-Wert aktualisieren
        if auto_render == True:
            # Aktualisiere den JSON-Wert für AutoRender auf True
            print("True")
            config["render_active"] = True
        else:
            print("False")
            # Aktualisiere den JSON-Wert für AutoRender auf False
            config["render_active"] = False

        # Aktualisiere den JSON-Wert für MinConfidence
        print("self Confidence:", min_confidence)
        config["min_confidence_threshold"] = min_confidence
        config["scene_accuracy"] = accuray_scene
        config["default_display_size"] = display_size
        

        # Speichere die aktualisierten Einstellungen in der JSON-Datei
        # with open(os.getcwd()+'/System/config.json', 'w') as json_file:
        #     json.dump(self.config, json_file)
            
        with open(os.path.join(os.getcwd(),"System","config.json"), 'w') as config_file:
            json.dump(config, config_file, indent=4)

        self.show_video2()      
    
    def play_video(self):
        if video_path and not self.playing:
            self.playing = True
            threading.Thread(target=self.play_thread).start()
    
    def play_thread(self):    
        
        cap = cv2.VideoCapture(video_path)
        FrameNbr=0
        image_paths = [os.path.join("Objectdetection","GUI_Assets","strand.png"),os.path.join("Objectdetection","GUI_Assets","stadt.png"),os.path.join("Objectdetection","GUI_Assets","wald.png")]
        
        #tracker, class_list, class_list_scene, model, model2, detection_colors, frame_wid, frame_hyt, fps, video_length = OD.load_model_tracker(cap)
        model, model2, byte_tracker,  byte_tracker2, annotator, fps, video_length = OD.load_models(cap)
        
        #Load Scene Model
        model_scene, transform, class_names = SD.load_scene_model()        
        #scene = sceneDetection.runSceneDetection(video_path)
        
        
        accuracy_scene = config["scene_accuracy"]
        
        dict = {}  
        
        while self.playing:
            ret, frame = cap.read()           
            
            if not ret:
                self.playing = False
                break            

            FrameNbr = FrameNbr + 1 
            
            if FrameNbr == 1 or FrameNbr % 5 == 0:
                scene_pic_id = SD.get_scene_pic(frame, model_scene, transform)
                print("pred item", scene_pic_id)                  
            
            if FrameNbr == 1 or FrameNbr % accuracy_scene == 0:
                votes = SD.get_scene_votes(frame, model_scene, transform, class_names)
               #scene = OD.analyze_scene_from_video(frame, class_list_scene)
        
            dict = {}        
                       
            #dict, frame = OD.analyse_frame_object(tracker, class_list, class_list_scene, model, model2,detection_colors, frame_wid, frame_hyt, fps, frame, FrameNbr)
            frame, dict = OD.callback(frame, 0, model, model2, FrameNbr, fps)  
            
            scene_image_path = image_paths[scene_pic_id]
            scene_image = ImageTk.PhotoImage(Image.open(scene_image_path).resize((25, 25)))
            
            self.play_button.configure(image=scene_image) 
            self.play_button.configure(text="Please wait...")             
            self.play_button.configure(fg_color="#6CC5D8")
            self.play_button.configure(text_color_disabled="white")
            
            if self.settings_button.winfo_exists():
                self.settings_button.destroy()
                
            if hasattr(self, 'set_button2') and self.set_button2.winfo_exists():
            # Your code here
                self.set_button2.destroy()
                pass

            
            self.play_button.configure(state=tk.DISABLED)  
            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)     
            
            self.video_label2.config(image=img)
            self.video_label2.image = img
        
        scene = SD.resume_votes(class_names)   
        print("Die Scene ist:", scene)
       
        cap.release()
        print(video_length)
        atmoApi.create_atmo(scene, dict, video_path, video_length)
    
    def stop_video(self):
        self.playing = False
    
    def show_thumbnail(self):
        global video_path
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((700, 400))
            img = ImageTk.PhotoImage(img)
            
            
            if hasattr(self, 'video_label'):
                self.video_label2.config(image=img)
                self.video_label2.image = img
            else:
                self.video_label2 = tk.Label(self.video_frame, image=img)
                self.video_label2.image = img
                self.video_label2.pack()
            
        
        cap.release()

    def end_GUI(self,project_path):
        self.video_player.stop()
        self.video_player.destroy()
        self.video_frame.destroy()
        self.center_frame.destroy()
        self.choose_button.destroy()
        self.play_button.destroy()
              
        # Defining variables
        video_path = None
        self.playing = False
        
        # Create a frame to center the video frame
        center_frame = tk.Frame(self.root)
        center_frame.pack(expand=True)      
        
        self.icon_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","prufen.png")).resize((200, 200)))
        self.icon_label = tk.Label(center_frame, image=self.icon_image) 
        self.icon_label.pack()       
        
        self.text = tk.Label(center_frame, text="Perfect! Your file is ready", font=("Calibri Bold", 20) , padx=20, pady=10, fg="#0C1745")
        self.text.pack(pady=(0, 20))      
                
        chooseButton_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","ordner2.png")).resize((40, 30), Image.ANTIALIAS))
        self.choose_button = customtkinter.CTkButton(center_frame, text="go to folder", font=("Calibri Bold", 20), image=chooseButton_image, command=self.open_folder, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.choose_button.pack(pady=(20, 0))
        
        createSound_image = ImageTk.PhotoImage(Image.open(os.path.join("Objectdetection","GUI_Assets","sback.png")).resize((20, 20), Image.ANTIALIAS))   
        self.play_button = customtkinter.CTkButton(center_frame, text="back", font=("Calibri Bold", 20), image=createSound_image, command=self.back_to_menu, width=250, height=50, compound="left", hover_color="#6CC5D8", fg_color="#0C1745")
        self.play_button.pack(pady=(10, 5))
        self.play_button.configure(fg_color="grey") 
    
    def process_end(self, project_path):
        self.root.attributes('-topmost',True)
        self.root.attributes('-topmost',False)
        self.end_GUI(project_path)
        
    def open_folder(self):
        # Öffne den Ordner im Explorer
        folder_path = config['render_strings']['RENDER_FILE']
        if os.path.exists(folder_path):
            if platform.system() == "Windows":
                os.system(f'explorer {folder_path}')
            else:
                subprocess.run(["/usr/bin/open",folder_path])

    def back_to_menu(self):
        
        self.icon_label.destroy()
        self.text.destroy()
        self.choose_button.destroy()
        self.play_button.destroy()
        
        self.show_sound_generator_app()


if __name__ == "__main__":
    slider_label=0
    global video_path 
        
    root = tk.Tk()
    video_player = TkinterVideo(master=root)
    app = VideoPlayerApp(root, video_player)
    atmoApi = AtmoApi.AtmoApi(app)
    root.attributes('-topmost',True)
    root.mainloop()
    
    

              
   
