import requests
import random
import json
import os

with open(os.getcwd()+"/System/config.json") as config_file:
    config = json.load(config_file)

address = config['ipaddress']

def search_object_file(tags, max_duration):

    # lower chars and sort tags ascending in order to match the SQL tags table

    lower_tags = [x.lower() for x in tags]
    sorted_tags = ','.join(sorted(lower_tags))

    # filtered GET request and dict conversion!

    # Get only the first 20 object samples with equal or greater duration time
    # and return one random selected sample

    samples = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                           "&cat=object"
                           "&duration=gte."+str(max_duration)+
                           "&order=duration.asc"
                           "&limit=20", verify=False).json()
    
    if len(samples)!= 0:
        # choose random sample
        #print("################# Type(samples): " + str(type(samples)))
        #print(samples)
        sample = samples[random.randrange(len(samples))]
        
        return sample
    
    # if no sample with the specified time exists, 
    # the next best sample with a lower duration time is selected

    samples = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                           "&cat=object"
                           "&duration=lte."+str(max_duration)+
                           "&order=duration.desc"
                           "&limit=20", verify=False).json()
    
    if len(samples) != 0:
        return samples[0]
    
    # if no sample is found, an empty dict is returned

    return {}


def search_scene_file(tags, max_duration):

    # lower chars and sort tags ascending in order to match the SQL tags table

    lower_tags = [x.lower() for x in tags]
    sorted_tags = ','.join(sorted(lower_tags))

    # checking whether a sample with the tags exists

    x = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                     "&cat=scene"
                     "&limit=2", verify=False).json()
    
    if len(x) == 0:
        return []

    # Filtered GET request and dict conversion

    # Get only the first 20 scene ambisonic samples with equal or greater duration time
    # and return the next best sample

    x = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                     "&cat=scene"
                     "&channel=gte.4"
                     "&duration=gte."+str(max_duration)+
                     "&order=duration.asc"
                     "&limit=20", verify=False).json()

    # if no sample with the specified time exists, 
    # the next best sample with a lower duration time is selected

    if len(x) == 0:
        x = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                         "&cat=scene"
                         "&channel=gte.4"
                         "&duration=lte."+str(max_duration)+
                         "&order=duration.desc"
                         "&limit=20", verify=False).json()

    if len(x) != 0:
        return [x[0]]

    # if no ambisonic file is found, 3 randomly selected stereo files are selected

    x = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                     "&cat=scene"
                     "&channel=lte.2"
                     "&duration=gte."+str(max_duration)+
                     "&limit=20", verify=False).json()

    if len(x) >= 3:

        # return of 3 different randomly chosen samples

        index_list = random.sample(range(0, len(x)), 3)
        return [x[index_list[0]],x[index_list[1]],x[index_list[2]]]

    # If there are not enough samples whose duration is greater than the one you are looking for. 
    # 3 different samples are randomly selected from the complete list

    x = requests.get("https://"+address+"/api/rpc/getallsamples?t_v={"+sorted_tags+"}"
                     "&cat=scene"
                     "&channel=lte.2"
                     "&order=duration.desc"                       
                     "&limit=20", verify=False).json()

    index_list = random.sample(range(0, len(x)), 3)
    return [x[index_list[0]],x[index_list[1]],x[index_list[2]]]
    

if __name__ == "__main__":
    address = '139.6.76.107'
    print(search_object_file(['forest'],5))
    #search_scene_file(['driving','car'],2)