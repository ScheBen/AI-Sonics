from matplotlib import pyplot as plt
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.autograd import Variable
import torchvision
import shutil

from tqdm.auto import tqdm

import os

from sklearn.model_selection import train_test_split

import copy
import datetime 
import time

import seaborn as sns




def determineDevice():
    if torch.cuda.is_available(): return torch.device('cuda')
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available(): return torch.device('mps')
    else: return torch.device('cpu')





def print3x3(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(dataset.classes[label])
        plt.axis("off")
        #plt.imshow(img.squeeze(), cmap="gray")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()





def rm_r(path, confirm = True, root = True):
    pathContent = os.listdir(path)
    absPath = os.path.abspath(path)

    choice = ""

    if confirm:
        print("-----------------!!!WARNING!!!-----------------")
        print("You are about to RECURSIVELY DELETE: ")
        print(absPath)
        print("If you want to continue enter: YES ")
        print("If you want to abort, enter anything else")

        choice = input()
        
        if choice != "YES":
            print("Deletion aborted")
            return

    for path in pathContent:
        path = os.path.join(absPath, path)
        if os.path.isdir(path):
            rm_r(path, False ,False)
            os.rmdir(path)
        else: 
            os.remove(path)
    
    if root: os.rmdir(absPath)




def loadDatasetPaths(baseDir):
    absBase = os.path.abspath(baseDir)

    classList = os.listdir(absBase)
    #classNums = np.arange(len(classList))
    classNums = []
    for i in range(len(classList)):
        classNums.append(i)
    
    classes = dict(zip(classList, classNums))
    classesInv = dict(zip(classNums, classList))

    imgIndex = []
    imgLabels = []

    for imgClass in classList:
        classPath = os.path.join(absBase, imgClass)
        imgList =  os.listdir(classPath)
        for i in range(len(imgList)): imgList[i] = os.path.join(classPath, imgList[i])
        imgIndex += imgList
        imgLabels += [classes[imgClass]] * len(imgList)
        

    retVal = {
        "imgIndex": imgIndex,
        "imgLabels": imgLabels,
        "classes": classList,
        "classDict": classes,
        "classDictInv": classesInv,
    }

    return retVal




class ListDataset():
    def __init__(self, data, labels, classDict, transform=None, target_transform=None):

        self.classes = classDict

        self.imgIndex = []
        self.imgLabels = []

        self.imgIndex = data
        self.imgLabels = labels

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgLabels)

    def __getitem__(self, idx):
        label = self.imgLabels[idx]
        image = Image.open(self.imgIndex[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label







def preTransform(baseDir, suffix, transforms, normalize = True, delPreNormalize = False):
    absBase = os.path.abspath(baseDir)

    classList = os.listdir(absBase)

    classNums = []
    for i in range(len(classList)):
        classNums.append(i)
    
    classes = dict(zip(classList, classNums))

    imgIndexOrig = []
    imgIndexNew = []

    newAbsBase = absBase+"_"+suffix

    for imgClass in classList:
        classPathOrig = os.path.join(absBase, imgClass)
        classPathNew = os.path.join(newAbsBase, imgClass)
        
        os.makedirs(classPathNew)

        imgListOrig =  os.listdir(classPathOrig)
        imgListNew = [""]*len(os.listdir(classPathOrig))

        print("classPathNew:", classPathNew)
        
        for i in range(len(imgListOrig)):
            origFull = os.path.join(classPathOrig, imgListOrig[i])
            imgListNew[i] = os.path.join(classPathNew, imgListOrig[i])
            imgListOrig[i] = origFull

        imgIndexOrig += imgListOrig
        imgIndexNew += imgListNew
    
    for i in tqdm(range(len(imgIndexOrig))):
        pathOrig = imgIndexOrig[i]
        pathNew = imgIndexNew[i]
        
        image = Image.open(pathOrig)
        image = transforms(image)
        image.save(os.path.splitext(pathNew)[0]+".png","PNG")
    
    if normalize:
        _, imgIndexScaled = scale(newAbsBase, save = True, delSoruce = delPreNormalize)

        retVal = {
        "imgIndexOrig": imgIndexOrig,
        "imgIndexNew": imgIndexNew,
        "imgIndexScaled": imgIndexScaled,
        }
    else:
        retVal = {
        "imgIndexOrig": imgIndexOrig,
        "imgIndexNew": imgIndexNew,
        }
    
    return retVal





def scale(path, save = True, delSoruce = False):
    #Load Data
    newDataSetPaths = loadDatasetPaths(path)
    newDataSet = ListDataset(newDataSetPaths["imgIndex"], newDataSetPaths["imgLabels"], newDataSetPaths["classDictInv"], transform = transforms.ToTensor())
    newDataLoader = torch.utils.data.DataLoader(newDataSet, batch_size=len(newDataSet), shuffle=False)
    newDataTensor, labels = next(iter(newDataLoader))

    #Convert 2D image Tensor to 1D image ndarray
    npScaled = newDataTensor.numpy()
    npScaledFlat = npScaled.reshape(newDataTensor.shape[0], newDataTensor.shape[1], newDataTensor.shape[2]*newDataTensor.shape[3])

    #Calc mean per color
    npScaledMean0 = npScaledFlat[:,0].mean()
    npScaledMean1 = npScaledFlat[:,1].mean()
    npScaledMean2 = npScaledFlat[:,2].mean()
    print(f"      Mean: R: {npScaledMean0:.6f}, G: {npScaledMean1:.6f}, B: {npScaledMean2:.6f}")

    #Calc standard deviation per color
    npScaledStd0 = npScaledFlat[:,0].std()
    npScaledStd1 = npScaledFlat[:,1].std()
    npScaledStd2 = npScaledFlat[:,2].std()
    print(f"    StdDev: R: {npScaledStd0:.6f}, G: {npScaledStd1:.6f}, B: {npScaledStd2:.6f}")

    #Remove mean
    npScaledFlat[:,0] = npScaledFlat[:,0]-npScaledMean0
    npScaledFlat[:,1] = npScaledFlat[:,1]-npScaledMean1
    npScaledFlat[:,2] = npScaledFlat[:,2]-npScaledMean2

    #Normalize standard deviation
    npScaledFlat[:,0] /= npScaledStd0
    npScaledFlat[:,1] /= npScaledStd1
    npScaledFlat[:,2] /= npScaledStd2

    print(f"  New Mean: R: {npScaledFlat[:,0].mean():.6f}, G: {npScaledFlat[:,1].mean():.6f}, B: {npScaledFlat[:,2].mean():.6f}")
    print(f"New StdDev: R: {npScaledFlat[:,0].std():.6f}, G: {npScaledFlat[:,1].std():.6f}, B: {npScaledFlat[:,2].std():.6f}")

    #Convert 1D image ndarray to 2D image Tensor
    npScaled = npScaledFlat.reshape(newDataTensor.shape[0], newDataTensor.shape[1], newDataTensor.shape[2],newDataTensor.shape[3])
    newTensor = torch.from_numpy(npScaled)
    
    imgIndex = []
    if(save):
        #Save images
        absBase = os.path.abspath(path)
        newBase = absBase+"_Norm"
        labels = labels.numpy()

        #Create Folder structure
        for imgClass in newDataSetPaths["classDict"]:
            classPathNew = os.path.join(newBase, imgClass)
            os.makedirs(classPathNew)

        #Save
        for i in range(newTensor.shape[0]):
            newPath = os.path.join(newBase, newDataSetPaths["classDictInv"][labels[i]], os.path.split(newDataSetPaths["imgIndex"][i])[1])
            imgIndex.append(newPath)
            torchvision.utils.save_image(newTensor[i], newPath)

    if(delSoruce):
        rm_r(path)

    return newTensor, imgIndex

def scaleGray(path, save = True, delSoruce = False):
    #Load Data
    newDataSetPaths = loadDatasetPaths(path)
    newDataSet = ListDataset(newDataSetPaths["imgIndex"], newDataSetPaths["imgLabels"], newDataSetPaths["classDictInv"], transform = transforms.ToTensor())
    newDataLoader = torch.utils.data.DataLoader(newDataSet, batch_size=len(newDataSet), shuffle=False)
    newDataTensor, labels = next(iter(newDataLoader))

    #Convert 2D image Tensor to 1D image ndarray
    npScaled = newDataTensor.numpy()
    npScaledFlat = npScaled.reshape(newDataTensor.shape[0], newDataTensor.shape[1], newDataTensor.shape[2]*newDataTensor.shape[3])

    #Calc mean per color
    npScaledMean0 = npScaledFlat[:,0].mean()
    print(f"      Mean: {npScaledMean0:.6f}")

    #Calc standard deviation per color
    npScaledStd0 = npScaledFlat[:,0].std()
    print(f"    StdDev: {npScaledStd0:.6f}")

    #Remove mean
    npScaledFlat[:,0] = npScaledFlat[:,0]-npScaledMean0


    #Normalize standard deviation
    npScaledFlat[:,0] /= npScaledStd0


    print(f"  New Mean: R: {npScaledFlat[:,0].mean():.6f}")
    print(f"New StdDev: R: {npScaledFlat[:,0].std():.6f}")

    #Convert 1D image ndarray to 2D image Tensor
    npScaled = npScaledFlat.reshape(newDataTensor.shape[0], newDataTensor.shape[1], newDataTensor.shape[2],newDataTensor.shape[3])
    newTensor = torch.from_numpy(npScaled)
    
    imgIndex = []
    if(save):
        #Save images
        absBase = os.path.abspath(path)
        newBase = absBase+"_Norm"
        labels = labels.numpy()

        #Create Folder structure
        for imgClass in newDataSetPaths["classDict"]:
            classPathNew = os.path.join(newBase, imgClass)
            os.makedirs(classPathNew)

        #Save
        for i in range(newTensor.shape[0]):
            newPath = os.path.join(newBase, newDataSetPaths["classDictInv"][labels[i]], os.path.split(newDataSetPaths["imgIndex"][i])[1])
            imgIndex.append(newPath)
            torchvision.utils.save_image(newTensor[i], newPath)

    if(delSoruce):
        rm_r(path)

    return newTensor, imgIndex


def TrainTestSplitDLO(baseDir = 'Train_40x60', trainSize = 0.8, transforms = None, random_state=42, shuffle = True, stratify = True):
    entireDataset = loadDatasetPaths(baseDir)

    X_train, X_test, y_train, y_test = train_test_split(
        entireDataset["imgIndex"], 
        entireDataset["imgLabels"],
        test_size = 1-trainSize,
        train_size = trainSize,
        random_state = random_state,
        shuffle = shuffle,
        stratify = entireDataset["imgLabels"] if stratify else None
    )

    trainingSet = ListDataset(X_train, y_train, entireDataset["classDictInv"], transforms)
    testSet = ListDataset(X_test, y_test, entireDataset["classDictInv"], transforms)

    return trainingSet, testSet


def training_loop_ES(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, EShistory, device):
    batch_loss = []
    epoch_loss = []
    
    epoch_accuracy = []
    val_losses = []
    val_accu = []
    best_val_loss = float('Inf')

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):  
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for imgs, labels in tqdm(train_loader): 
            
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()# * imgs.size(0)
            batch_loss.append(loss.item())

            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]  
            correct += int((predicted == labels).sum())



        epoch_loss.append((running_loss/len(train_loader)))
        epoch_accuracy.append(100* correct / total)

        val_loss = 0
        accuracy=0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = model(images)
                val_loss += loss_fn(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        val_losses.append(val_loss/len(val_loader))
        val_accu.append(accuracy/len(val_loader))

        print("Epoch: {}/{} ".format(epoch+1, n_epochs),
            "Time: {:.2f}s ".format(time.time()-epoch_start_time),
            "Training Loss: {:.3f} ".format(epoch_loss[-1]),
            "Training Accu: {:.2f}% ".format(epoch_accuracy[-1]),
            "Val Loss: {:.3f} ".format(val_losses[-1]),
            "Val Accu: {:.2f}%".format(100*val_accu[-1]))

        print( '  Epoch took %6.2fs, toal %8.2fs' % (time.time()-epoch_start_time, time.time()-start_time) )
        
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            counter=0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            counter+=1
            print('Validation loss has not improved since: {:.3f}..'.format(best_val_loss), 'Count: ', str(counter))
            if counter >= EShistory:
                #print('Early Stopping now!!')
                #model.load_state_dict(best_model_wts)
                #break
                print("Stop Early? [Y/N]")
                answer = input()
                if answer == "Y" or answer == "y":
                    print('Early Stopping now!!')
                    model.load_state_dict(best_model_wts)
                    break

                print("Load best parameters upto now? [Y/N]")
                answer = input()
                if answer == "Y" or answer == "y":
                    model.load_state_dict(best_model_wts)
                
                print("Reduce counter to continue multiple epochs? [Number]")
                answer = input()
                if answer > 0:
                    counter -= answer


            
            
    return batch_loss, epoch_loss

def training_loop_ES_Stack(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, EShistory, device):
    batch_loss = []
    epoch_loss = []
    
    epoch_accuracy = []
    val_losses = []
    val_accu = []
    best_val_loss = float('Inf')

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):  
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for imgs, labels in tqdm(train_loader): 
            
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()# * imgs.size(0)
            batch_loss.append(loss.item())

            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]  
            correct += int((predicted == labels).sum())



        epoch_loss.append((running_loss/len(train_loader)))
        epoch_accuracy.append(100* correct / total)

        val_loss = 0
        accuracy=0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = model(images)
                val_loss += loss_fn(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        val_losses.append(val_loss/len(val_loader))
        val_accu.append(accuracy/len(val_loader))

        print("Epoch: {}/{} ".format(epoch+1, n_epochs),
            "Time: {:.2f}s ".format(time.time()-epoch_start_time),
            "Training Loss: {:.3f} ".format(epoch_loss[-1]),
            "Training Accu: {:.2f}% ".format(epoch_accuracy[-1]),
            "Val Loss: {:.3f} ".format(val_losses[-1]),
            "Val Accu: {:.2f}%".format(100*val_accu[-1]))

        print( '  Epoch took %6.2fs, toal %8.2fs' % (time.time()-epoch_start_time, time.time()-start_time) )
        
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            counter=0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            counter+=1
            print('Validation loss has not improved since: {:.3f}..'.format(best_val_loss), 'Count: ', str(counter))
            if counter >= EShistory:
                print("Stop Early? [Y/N]")
                answer = input()
                if answer == "Y" or answer == "y":
                    print('Early Stopping now!!')
                    model.load_state_dict(best_model_wts)
                    break

                print("Load best parameters upto now? [Y/N]")
                answer = input()
                if answer == "Y" or answer == "y":
                    model.load_state_dict(best_model_wts)

                
                print("Reduce counter to continue multiple epochs? [Number]")
                answer = input()
                if answer > 0:
                    counter -= answer

            
            
    return batch_loss, epoch_loss





def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    batch_loss = []
    epoch_loss = []
    for epoch in range(1, n_epochs + 1):  
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader): 
            
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()# * imgs.size(0)
            batch_loss.append(loss.item())

            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]  
            correct += int((predicted == labels).sum())
            
        if True:#epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                running_loss / len(train_loader)))
            epoch_loss.append((running_loss/len(train_loader)))
            print("Accuracy: {:.2f} %".format(100* correct / total))

            
            
    return batch_loss, epoch_loss





def validate(model, train_loader, val_loader, device):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in tqdm(loader):

                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                
                outputs = model(imgs)
                
                _, predicted = torch.max(outputs, dim=1) # <2>
                
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy {}: {:.2f} %".format(name , 100 * correct / total))





def normalizeColumns(matrix):
    colSum = matrix.sum(axis=0)
    return  (matrix/colSum)*100





# func to normalize the rows of a matrix
def normalizeRows(matrix):
    
    for r in range(matrix.shape[0]):
        rowsum = 0
        for c in range(matrix.shape[0]):
            rowsum += matrix[r][c]

        for c in range(matrix.shape[0]):
            matrix[r][c] = matrix[r][c] * 100 / rowsum

    return  matrix






#function to plot the confusion matrix for a models prediction on the test and train set
def plotConfusionMatrix(trainSet, testSet, model):

    classes = trainSet.classes.values();
    print(classes)
    nClasses = len(classes)

    confusionMatrixTrain = np.zeros([nClasses,nClasses])
    confusionMatrixTest = np.zeros([nClasses,nClasses])

    #Dataloaders
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=len(trainSet), shuffle=False)  
    val_loader = torch.utils.data.DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(val_loader))

    #make predictions on training set
    y_train_pred_raw = model(X_train)
    #make predictions on test set
    y_test_pred_raw = model(X_test)

    _, y_train_pred = torch.max(y_train_pred_raw, dim=1)
    _, y_test_pred = torch.max(y_test_pred_raw, dim=1)

    #compute confusion matrix for Training set
    for i in range(len(y_train)):
        
        confusionMatrixTrain[y_train[i].item(), y_train_pred[i].item()] += 1
    #compute confusion matrix for Test set
    for i in range(len(y_test)):
        confusionMatrixTest[y_test[i].item(), y_test_pred[i].item()] += 1

    #compute accuracy  for Training set
    accuracyScoreTrain = int((y_train_pred == y_train).sum()) / len(y_train)
    #compute accuracy  for Test set
    accuracyScoreTest = int((y_test_pred == y_test).sum()) / len(y_test)

    # plot the confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5)) 
    tickLabels = classes

    confusionMatrixPlotTrain = sns.heatmap(normalizeRows(confusionMatrixTrain), annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".1f", ax=ax1)
    confusionMatrixPlotTrain.set_title(f"Train Accuracy: {accuracyScoreTrain*100}%")
    confusionMatrixPlotTrain.set_xlabel("Predicted")
    confusionMatrixPlotTrain.set_ylabel("Actual")

    confusionMatrixPlotTest = sns.heatmap(normalizeRows(confusionMatrixTest), annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".1f", ax=ax2)
    confusionMatrixPlotTest.set_title(f"Val. Accuracy: {accuracyScoreTest*100}%")
    confusionMatrixPlotTest.set_xlabel("Predicted")
    confusionMatrixPlotTest.set_ylabel("Actual")

    confusionMatrixTest.sum(axis=0)

#function to plot the confusion matrix for a models prediction on the test and train set
def plotConfusionMatrixTmp(trainSet, testSet, model):

    classes = trainSet.classes.values();
    print(classes)
    nClasses = len(classes)

    confusionMatrixTrain = np.zeros([nClasses,nClasses])
    confusionMatrixTest = np.zeros([nClasses,nClasses])

    #Dataloaders
    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=len(trainSet), shuffle=False)  
    val_loader = torch.utils.data.DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(val_loader))

    #make predictions on training set
    y_train_pred_raw = model(X_train)
    #make predictions on test set
    y_test_pred_raw = model(X_test)

    _, y_train_pred = torch.max(y_train_pred_raw, dim=1)
    _, y_test_pred = torch.max(y_test_pred_raw, dim=1)

    #compute confusion matrix for Training set
    for i in range(len(y_train)):
        
        confusionMatrixTrain[y_train[i].item(), y_train_pred[i].item()] += 1
    #compute confusion matrix for Test set
    for i in range(len(y_test)):
        confusionMatrixTest[y_test[i].item(), y_test_pred[i].item()] += 1

    #compute accuracy  for Training set
    accuracyScoreTrain = int((y_train_pred == y_train).sum()) / len(y_train)
    #compute accuracy  for Test set
    accuracyScoreTest = int((y_test_pred == y_test).sum()) / len(y_test)

    # plot the confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5)) 
    tickLabels = classes

    confusionMatrixPlotTrain = sns.heatmap(normalizeRows(confusionMatrixTrain), annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".1f", ax=ax1)
    confusionMatrixPlotTrain.set_title(f"Train Accuracy: {accuracyScoreTrain*100}%")
    confusionMatrixPlotTrain.set_xlabel("Predicted")
    confusionMatrixPlotTrain.set_ylabel("Actual")

    confusionMatrixPlotTest = sns.heatmap(normalizeRows(confusionMatrixTest), annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".1f", ax=ax2)
    confusionMatrixPlotTest.set_title(f"Val. Accuracy: {accuracyScoreTest*100}%")
    confusionMatrixPlotTest.set_xlabel("Predicted")
    confusionMatrixPlotTest.set_ylabel("Actual")

    confusionMatrixTest.sum(axis=0)

#Split RGB image in R,G,B Images and store as greyscale
def splitChannelsGS(oPath, delSource = False):
    
    absBase = os.path.abspath(oPath)
    newBase = absBase+"_GS"

    newDataSetPaths = loadDatasetPaths(absBase)
    
    #Create Folder structure
    for imgClass in newDataSetPaths["classDict"]:
        classPathNew = os.path.join(newBase, imgClass)
        os.makedirs(classPathNew)

    imgIndex = []
    for path in tqdm(newDataSetPaths["imgIndex"]):
        newPath = path.replace(absBase, newBase)
        imgIndex.append(newPath)
        img = Image.open(path)
        imgSplit = Image.Image.split(img)

        imgR = imgSplit[0].convert('L')
        imgG = imgSplit[1].convert('L')
        imgB = imgSplit[2].convert('L')
        pathComponents = os.path.splitext(newPath)
        imgR.save(pathComponents[0]+ "R" +pathComponents[1])
        imgG.save(pathComponents[0]+ "G" +pathComponents[1])
        imgB.save(pathComponents[0]+ "B" +pathComponents[1])


    if(delSource):
        rm_r(path)
    
    return imgIndex