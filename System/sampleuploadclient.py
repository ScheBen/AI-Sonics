import paramiko
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import requests
import re


ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

#region
login = ""
pw = ""
rootpw = ""
address = "139.6.76.107"

auth_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoidG9kb19zYW1wbGUifQ.J_5MOzG9-tbW4w7y8XR5jMVX_6MPuWDpsO4MQYvEs4k'

#endregion

class GUI(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title("Sample Upload Client")
        self.geometry("450x600")

        self.file_location = ""


        self.grid_columnconfigure(0, weight = 0)

        self.frame_1 = ctk.CTkFrame(master=self)
        self.frame_1.grid(row = 0, column = 0, rowspan = 30, sticky = "nsew", padx = 10, pady = 10)        

        self.label_1 = ctk.CTkLabel(master=self.frame_1, justify=ctk.LEFT, text = "Sample Upload Client")
        self.label_1.grid(row = 0, columnspan = 3)

        self.line_1 = ctk.CTkFrame(master = self.frame_1, height = 4, fg_color = "#1a5594", corner_radius = 0)
        self.line_1.grid(row = 1, column = 0, columnspan = 3, pady = 5, padx = 5, sticky ="nswe")


        self.search_text = ctk.CTkTextbox(master=self.frame_1, width = 200, height = 30)
        self.search_text.grid(row = 2, column = 0, padx = 5, pady = 5)
        self.search_text.insert("0.0", "")

        self.search_button = ctk.CTkButton(master = self.frame_1,
                                     text = "Search Sample",
                                     command = self.search_file,
                                     width = 200)
        self.search_button.grid(row = 2, column = 2, padx = 5, pady = 5)


        self.line_2 = ctk.CTkFrame(master = self.frame_1, height = 4, fg_color = "#1a5594", corner_radius = 0)
        self.line_2.grid(row = 3, column = 0, columnspan = 3, pady = 5, padx = 5, sticky ="nswe")


        self.description_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Description")
        self.description_label.grid(row = 4, column = 0, padx = 5, pady = 5)

        self.description_text = ctk.CTkTextbox(master = self.frame_1, width = 200, height = 30)
        self.description_text.insert("0.0", "")
        self.description_text.grid(row = 4, column = 2, padx = 5, pady = 5)


        self.duration_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Duration")
        self.duration_label.grid(row = 6, column = 0, padx = 5, pady = 5)

        self.duration_text = ctk.CTkTextbox(master = self.frame_1, width = 200, height = 30)
        self.duration_text.insert("0.0", "")
        self.duration_text.grid(row = 6, column = 2, padx = 5, pady = 5)


        self.bitdepth_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Bit Depth")
        self.bitdepth_label.grid(row = 8, column = 0, padx = 5, pady = 5)

        self.bitdepth_option = ctk.CTkOptionMenu(master = self.frame_1, values = ["16","24", "32"], width = 200)
        self.bitdepth_option.grid(row = 8, column = 2, padx = 5, pady = 5)


        self.samplerate_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Samplerate")
        self.samplerate_label.grid(row = 10, column = 0, padx = 5, pady = 5)

        self.samplerate_option = ctk.CTkOptionMenu(master = self.frame_1, values = ["44100", "48000", "96000"], width = 200)
        self.samplerate_option.grid(row = 10, column = 2, padx = 5, pady = 5)


        self.channel_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Channel")
        self.channel_label.grid(row = 14, column = 0, padx = 5, pady = 5)

        self.channel_text = ctk.CTkTextbox(master = self.frame_1, width = 200, height = 30)
        self.channel_text.insert("0.0", "")
        self.channel_text.grid(row = 14, column = 2, padx = 5, pady = 5)


        self.category_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Category")
        self.category_label.grid(row = 16, column = 0, padx = 5, pady = 5)

        self.category_option = ctk.CTkOptionMenu(master = self.frame_1, values = ["scene", "object"], width = 200)
        self.category_option.grid(row = 16, column = 2, padx = 5, pady = 5)


        self.tag_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Tags")
        self.tag_label.grid(row = 18, column = 0, padx = 5, pady = 5)

        self.tag_text = ctk.CTkTextbox(master = self.frame_1, width = 200, height = 30)
        self.tag_text.insert("0.0", "")
        self.tag_text.grid(row = 18, column = 2, padx = 5, pady = 5)
        

        self.upload_button = ctk.CTkButton(master = self.frame_1,
                                     text = "Upload",
                                     command = self.upload_file)
        self.upload_button.grid(row = 20, column = 0, columnspan = 3, padx = 5, pady = 5, sticky = "nsew")


        self.account_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = "Account")
        self.account_label.grid(row = 22, column = 0, padx = 5, pady = 5)

        self.account_label = ctk.CTkLabel(master = self.frame_1, justify = ctk.LEFT, text = login)
        self.account_label.grid(row = 22, column = 2, padx = 5, pady = 5)

        self.line_3 = ctk.CTkFrame(master = self.frame_1, height = 4, fg_color = "#1a5594", corner_radius = 0)
        self.line_3.grid(row = 24, column = 0, columnspan = 3, pady = 5, padx = 5, sticky ="nswe")

        #self.info_frame = ctk.CTkFrame(master = self.frame_1)
        #self.info_frame.grid(row = 26, column = 0, columnspan = 3, rowspan = 1, sticky ="nswe")

        self.info_Label = ctk.CTkLabel(master=self.frame_1, justify=ctk.LEFT, text = "",font=("Roboto Medium",-16))
        self.info_Label.grid(row = 26, column = 0, columnspan = 3, rowspan = 3, sticky ="nswe")

    def search_file(self):

        display = filedialog.askopenfile(title = "Select a file", filetypes = [("Wav Files", "*.wav"),("MP3 Files", "*.mp3")])

        if display != None:
            self.search_text.delete("0.0","end")
            self.search_text.insert("0.0",display.name)
            self.file_location = display.name
            self.update()

        

    def upload_file(self):
        
        self.info_Label.configure(text="Loading", text_color = "green")
        self.update()

        tag_arr = re.split(r' , |, | ,|,| ',self.tag_text.get("0.0","end"))

        file_tags = []

        for i, tagname in enumerate(tag_arr):
            file_tags.append({"tagid":"","name":tagname.replace("\n","")})

        format = self.file_location[self.file_location.rfind(".")+1:]
        

        if len(self.file_location) <= 3:
            return

        #   Get the last 5 samples, in order to get the last sampleid

        samples = requests.get("https://"+address+"/api/samples"
                           "?order=sampleid.desc"
                           "&limit=5", verify=False).json()
    
        tags = requests.get("https://"+address+"/api/tags", verify=False).json()
        
        if len(samples) > 0:
            next_sample_id = int(samples[0]['sampleid'])+1
        else:
            next_sample_id = 1

        #   add sample to database

        requests.post('https://'+address+'/api/samples',
                                headers = {'Authorization': 'Bearer '+auth_key},
                                json = {"sampleid" : next_sample_id,
                                        "description": self.description_text.get("0.0", "end").replace("\n",""),
                                        "duration": self.duration_text.get("0.0","end").replace("\n",""),
                                        "bitdepth": self.bitdepth_option.get(),
                                        "samplerate": self.samplerate_option.get(),
                                        "format": format,
                                        "channel": self.channel_text.get("0.0", "end").replace("\n",""),
                                        "catid": 1 if self.category_option.get() == 'scene' else 2,
                                        "path": "https://"+address+"/"+format+"/"+str(next_sample_id)+"."+format}, verify=False) 

        #   Add tag and sample relation

        next_tag_id = len(tags) + 1

        for file_tag in file_tags:
            found = False

            for tag in tags:
            
                if tag['name'] == file_tag['name']:
                    file_tag['tagid'] = tag['tagid']
                    found = True

            #   add the actual tag if it doesn't exist in the database

            if found == False:
                file_tag['tagid'] = next_tag_id
                requests.post('https://'+address+'/api/tags',
                                headers = {'Authorization': 'Bearer '+auth_key},
                                json = {"tagid":file_tag['tagid'],"name": file_tag['name']}, verify=False)     
                next_tag_id += 1
            
            requests.post('https://'+address+'/api/tagsrelation',
                                headers = {'Authorization': 'Bearer '+auth_key},
                                json = {"tagid":file_tag['tagid'],"sampleid": next_sample_id}, verify=False)   
            
            
        #   open ssh connection

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=address,username=login,password=pw)

        #   open sftp connection and upload file

        ftp_client = ssh_client.open_sftp()
        ftp_client.put(self.file_location,'Dokumente/'+str(next_sample_id)+"."+format)
        ftp_client.close()

        #   move file to correct directory and close command

        stdin, stdout, stderr = ssh_client.exec_command("sudo -S -p '' mv Dokumente/"+str(next_sample_id)+"."+format+" /var/www/html/"+format+"/"+str(next_sample_id)+"."+format)
        stdin.write(rootpw+"\n")
        stdin.flush()
        stdin.close()

        self.info_Label.configure(text="saved to sampleid "+str(next_sample_id), text_color = "white")
        self.update()
        

if __name__ == "__main__":

    app = GUI()
    app.mainloop()

    


    