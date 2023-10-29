import os
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


model_path = os.path.join(os.getcwd(),"Scenedetection","MAP_FinalModel.pth")
votes = [0, 0, 0]

################################################ The Net ####################################################

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
        #print(x.size()) # Debugging Line
        x = x.view(-1, 32 * 28 * 51)
        x = F.elu(self.fc0(self.dropout0(self.fc0_bn(x))))
        x = F.elu(self.fc1(self.dropout1(x)))
        x = F.elu(self.fc2(x))
        return x


############################################## Load Model ######################################################

def load_scene_model():
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ["beach", "city", "forest"]
        
    
    return model, transform, class_names


############################################## Predict Frames #################################################    
    
def get_scene_votes(frame, model, transform, class_names):   
        resized_frame = cv2.resize(frame, (426, 240))
        frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)        
        image = transform(frame)

        with torch.no_grad():
            outputs = model(image.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            
        votes[predicted.item()] += 1
        
        return votes
        
############################################## Evaluation of Predictions #################################################     
def resume_votes(class_names):
    
    for i, class_name in enumerate(class_names):
        print(f"votes for {class_name}: {votes[i]}")

    max_votes = max(votes)
    max_class = [class_names[i] for i, v in enumerate(votes) if v == max_votes]

    result = max_class[0]

    print("The model has the most votes for the class:", result)
    print(max_class[0])

    return(max_class[0])

############################################## Predict Frame for Scene Thumbnail #################################################

def get_scene_pic(frame, model, transform):
    pred = ""
    
    resized_frame = cv2.resize(frame, (426, 240))
    frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)        
    image = transform(frame)

    with torch.no_grad():
        out= model(image.unsqueeze(0))
        _, pred = torch.max(out.data, 1)
    
    return pred.item()
