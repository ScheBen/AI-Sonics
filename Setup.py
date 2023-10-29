import os
import json
import sys
import subprocess
import platform
import venv
from time import sleep

p_system = platform.system()

#   Search for the Reaper.exe path
r_path =""
render_format=""
if p_system == "Darwin":
    #   "/Applications/REAPER.app/Contents/MacOS/REAPER"
    render_format = "FVAX"
    for root, dirs, files in os.walk("/Applications"):
        for name in files:
            if name == "REAPER":
                # Absolute Path reaper
                r_path = os.path.abspath(os.path.join(root, name))
                
                break
        else:
            continue
        break
    #r_path = os.path.join("","Applications","REAPER")
else:
    render_format =" FMW"
    drive_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    existing_drives = ['%s:'%d for d in drive_names if os.path.exists('%s:'%d)]
    drives = tuple(existing_drives)
    for d_name in drives:
        for root, dirs, files in os.walk(d_name):
            for name in files:
                if name == "reaper.exe":
                    # Absolute Path reaper
                    r_path = os.path.abspath(os.path.join(root, name))
                
                    break
            else:
                continue
            break

#   Load Json Config to dict

with open(os.path.join(os.getcwd(),"System","config.json"), "r") as config_file:
        config = json.load(config_file)

config["reaper_dir"] = r_path
config["render_strings"]["RENDER_FORMAT"] = render_format

#   Overwrite Json Config

with open(os.path.join(os.getcwd(),"System","config.json"), "w") as config_file:
    json.dump(config, config_file, indent=4)

#   Install required Packages from the Requirement.txt

dir = os.path.join(os.getcwd(), "aisonics_environment")
venv.create(dir, with_pip=True)
reaper_proc = subprocess.Popen(r_path)
sleep(2)

if p_system == "Darwin":
    subprocess.run(["pip","install","python-reapy"])
    subprocess.run(["python","-c","import reapy; reapy.configure_reaper()"]) 
    subprocess.run(["bin/pip", "install", "-r", os.path.abspath(os.path.join("System","requirements.txt"))], cwd=dir)
elif p_system == "Windows":
    subprocess.run(["py","-m","pip","install","python-reapy"])
    subprocess.run(["py","-c","import reapy; reapy.configure_reaper()"])
    subprocess.run([os.path.join(dir, "Scripts", "pip.exe"),"install","-r",os.path.abspath(os.path.join("System","requirements.txt"))])
    
reaper_proc.kill()