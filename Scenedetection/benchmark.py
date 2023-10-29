from matplotlib import pyplot as plt
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import os 

from tqdm.auto import tqdm

from PIL import Image

from sklearn.model_selection import train_test_split

import MAPutils as MAP

import seaborn as sns

def TrainTestSplitMulti(baseDir = 'Train', trainSize = 0.8, transforms = None, random_state=42, shuffle = True, stratify = True):
    entireDataset = MAP.loadDatasetPaths(baseDir)

    X_train, X_test, y_train, y_test = train_test_split(
        entireDataset["imgIndex"], 
        entireDataset["imgLabels"],
        test_size = 1-trainSize,
        train_size = trainSize,
        random_state = random_state,
        shuffle = shuffle,
        stratify = entireDataset["imgLabels"] if stratify else None
    )

    trainingSet = MultiListDataset(X_train, y_train, entireDataset["classDictInv"], transforms)
    testSet = MultiListDataset(X_test, y_test, entireDataset["classDictInv"], transforms)

    return trainingSet, testSet

class MultiListDataset():
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
            imgSplit = Image.Image.split(image)

            imgR = imgSplit[0].convert('L')
            imgG = imgSplit[1].convert('L')
            imgB = imgSplit[2].convert('L')
    
            toTensor = transforms.Compose([transforms.ToTensor()])
    
            imgR = toTensor(imgR)
            imgG = toTensor(imgG)
            imgB = toTensor(imgB)
            
        if self.target_transform:
            label = self.target_transform(label)
        

        return imgR, imgG, imgB, label, self.imgIndex[idx]


def predictVote(model, images, labels):
    
    outputsR = model(images[0])
    outputsG = model(images[1])
    outputsB = model(images[2])

    outrn = outputsR.numpy()
    outgn = outputsG.numpy()
    outbn = outputsB.numpy()

    outSum = (outrn + outgn +outbn)/3

    return torch.tensor(outSum), outputsR, outputsG, outputsB


import string


def benchmarkMulti(model, testDataPath, develop = False, split = 0.7):

    #Inplace Tranformations
    transform = transforms.Compose(
    [
        transforms.Resize(size = (426,240)),
    ]
    )

    #Load data
    if develop:
        _, testSet = TrainTestSplitMulti(
            baseDir = testDataPath, 
            trainSize = split,
            transforms = transform,
            random_state=42, 
            shuffle = True, 
            stratify = False)
    else:
        dataset = MAP.loadDatasetPaths(testDataPath)
        testSet = MultiListDataset(dataset["imgIndex"], dataset["imgLabels"], dataset["classDictInv"], transform)

    samples = torch.utils.data.DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    
    model.eval()

    accuracy = 0
    accuracyR = 0
    accuracyG = 0
    accuracyB = 0
    loss = 0
    lossR = 0
    lossG = 0
    lossB = 0

    loss_fn = nn.CrossEntropyLoss() 

    with torch.no_grad(): 
        r, g, b, labels, paths = next(iter(samples))
        prediction, predictionR, predictionG, predictionB = predictVote(model, [r, g, b], labels)
        
        #Vote
        #Loss
        loss =+ loss_fn(prediction, labels)
        #Predicted class
        ps = torch.exp(prediction)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        #Accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor))

        #R
        lossR =+ loss_fn(predictionR, labels)
        psR = torch.exp(predictionR)
        top_pR, top_classR = psR.topk(1, dim=1)
        equalsR = top_classR == labels.view(*top_classR.shape)
        accuracyR += torch.mean(equalsR.type(torch.FloatTensor))

        #G
        lossG =+ loss_fn(predictionG, labels)
        psG = torch.exp(predictionG)
        top_pG, top_classG = psG.topk(1, dim=1)
        equalsG = top_classG == labels.view(*top_classG.shape)
        accuracyG += torch.mean(equalsG.type(torch.FloatTensor))

        #B
        lossB =+ loss_fn(predictionB, labels)
        psB = torch.exp(predictionB)
        top_pB, top_classB = psB.topk(1, dim=1)
        equalsB = top_classB == labels.view(*top_classB.shape)
        accuracyB += torch.mean(equalsB.type(torch.FloatTensor))

    loss = loss / len(samples)
    lossR = lossR / len(samples)
    lossG = lossG / len(samples)
    lossB = lossB / len(samples)
    lossRGB = (lossR + lossG + lossB) / (3 * len(samples))


    classes = testSet.classes.values();
    nClasses = len(classes)


    # plot the confusion matrices
    confusionMatrix = np.zeros([nClasses,nClasses])
    confusionMatrixR = np.zeros([nClasses,nClasses])
    confusionMatrixG = np.zeros([nClasses,nClasses])
    confusionMatrixB = np.zeros([nClasses,nClasses])

    for i in range(len(prediction)):
        confusionMatrix[labels[i].item(), top_class[i].item()] += 1
        confusionMatrixR[labels[i].item(), top_classR[i].item()] += 1
        confusionMatrixG[labels[i].item(), top_classG[i].item()] += 1
        confusionMatrixB[labels[i].item(), top_classB[i].item()] += 1


    fig, (ax) = plt.subplots(2, 2, figsize=(16 ,14)) 
    tickLabels = classes

    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrix, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[0][0])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100*accuracy} % \n Loss: {loss}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Voting)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #R
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixR, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[0][1])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100*accuracyR} % \n Loss: {lossR}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Red)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #G
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixG, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[1][0])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100 * accuracyG} % \n Loss: {lossG}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Green)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #B
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixB, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[1][1])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100 * accuracyB} %s \n Loss: {lossB}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Blue)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    print(f"LossRGB: {lossRGB}")


    #fig.suptitle(f"Number of Samples {len(testSet)})")

    #determine wrongly classified images (only Vote)
    index = np.invert(equals.numpy().reshape(len(equals)))
    paths = np.array(paths)[index]

    wrongClasses = []
    tpc = top_class.numpy().reshape(len(equals))
    classes = tpc[index]
    
    rightClasses = []
    lbls = labels.view(*top_class.shape).numpy().reshape(len(equals))
    classes1 = lbls[index]

    wrongPredictions = []
    predsAll = prediction.numpy()
    preds = predsAll[index]

    for i in range(len(classes)):
        wrongClasses.append(testSet.classes.get(classes[i]))
        rightClasses.append(testSet.classes.get(classes1[i]))
        tmp = preds[i]
        tmp -= tmp.min()
        tmp /= tmp.sum()

        wrongPredictions.append(tmp)



    return accuracy, confusionMatrix, [paths, wrongClasses, rightClasses, wrongPredictions, index], preds



import math
def printMissclassified(paths, wrongClasses, rightClasses, wrongPredictions):

    nImages = len(wrongClasses)

    if(nImages < 4):
        cols = nImages
        rows = 1
    else:
        cols = 3
        rows = int(math.ceil(nImages/3))

    figure = plt.figure(figsize=(3*cols, 4*rows))
    transfrom = transforms.Compose([transforms.ToTensor()])
    
    for i in range(1, nImages + 1):
        image = Image.open(paths[i-1])
        image = transfrom(image)
        figure.add_subplot(rows, cols, i)
        titleString = "Label: %s \n Prediction: %s \n p(Rock): %6.2f %% \n p(Paper): %6.2f %% \n p(Scissors): %6.2f %% \n p(Rest): %6.2f %%" % (rightClasses[i-1], wrongClasses[i-1], wrongPredictions[i-1][2]*100, wrongPredictions[i-1][0]*100, wrongPredictions[i-1][3]*100, wrongPredictions[i-1][1]*100)
        #plt.title(f"Label: {rightClasses[i-1]} \n Prediction: {wrongClasses[i-1]} \n Probability Rock: {wrongPredictions[i-1][2]*100} % \n Probability Paper: {wrongPredictions[i-1][0]*100} % \n Probability Scissors: {wrongPredictions[i-1][3]*100} % \n Probability Rest: {wrongPredictions[i-1][1]*100} %")
        plt.title(titleString)
        plt.axis("off")
        #plt.imshow(img.squeeze(), cmap="gray")
        plt.imshow(image.permute(1, 2, 0))
    plt.show()



    ##################################################Stacking

def genStackingData(model, testDataPath, train, split = 0.7):

    #Inplace Tranformations
    transform = transforms.Compose(
    [
        transforms.Resize(size = (426,240)),
    ]
    )

    #Load data
    
    trainSet, testSet = TrainTestSplitMulti(
        baseDir = testDataPath, 
        trainSize = split,
        transforms = transform,
        random_state=42, 
        shuffle = True, 
        stratify = False)

    if(train):
        samples = torch.utils.data.DataLoader(trainSet, batch_size=len(testSet), shuffle=False)
    else:
        samples = torch.utils.data.DataLoader(testSet, batch_size=len(testSet), shuffle=False)
    model.eval()

    accuracy = 0
    accuracyR = 0
    accuracyG = 0
    accuracyB = 0
    loss = 0
    lossR = 0
    lossG = 0
    lossB = 0

    loss_fn = nn.CrossEntropyLoss() 

    with torch.no_grad(): 
        r, g, b, labels, paths = next(iter(samples))
        prediction, predictionR, predictionG, predictionB = predictVote(model, [r, g, b], labels)
        
        #Vote
        #Loss
        loss =+ loss_fn(prediction, labels)
        #Predicted class
        ps = torch.exp(prediction)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        #Accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor))

        #R
        lossR =+ loss_fn(predictionR, labels)
        psR = torch.exp(predictionR)
        top_pR, top_classR = psR.topk(1, dim=1)
        equalsR = top_classR == labels.view(*top_classR.shape)
        accuracyR += torch.mean(equalsR.type(torch.FloatTensor))

        #G
        lossG =+ loss_fn(predictionG, labels)
        psG = torch.exp(predictionG)
        top_pG, top_classG = psG.topk(1, dim=1)
        equalsG = top_classG == labels.view(*top_classG.shape)
        accuracyG += torch.mean(equalsG.type(torch.FloatTensor))

        #B
        lossB =+ loss_fn(predictionB, labels)
        psB = torch.exp(predictionB)
        top_pB, top_classB = psB.topk(1, dim=1)
        equalsB = top_classB == labels.view(*top_classB.shape)
        accuracyB += torch.mean(equalsB.type(torch.FloatTensor))

    loss = loss / len(samples)
    lossR = lossR / len(samples)
    lossG = lossG / len(samples)
    lossB = lossB / len(samples)
    lossRGB = (lossR + lossG + lossB) / (3 * len(samples))


    classes = testSet.classes.values();
    nClasses = len(classes)


    # plot the confusion matrices
    confusionMatrix = np.zeros([nClasses,nClasses])
    confusionMatrixR = np.zeros([nClasses,nClasses])
    confusionMatrixG = np.zeros([nClasses,nClasses])
    confusionMatrixB = np.zeros([nClasses,nClasses])

    for i in range(len(prediction)):
        confusionMatrix[labels[i].item(), top_class[i].item()] += 1
        confusionMatrixR[labels[i].item(), top_classR[i].item()] += 1
        confusionMatrixG[labels[i].item(), top_classG[i].item()] += 1
        confusionMatrixB[labels[i].item(), top_classB[i].item()] += 1


    fig, (ax) = plt.subplots(2, 2, figsize=(16 ,14)) 
    tickLabels = classes

    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrix, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[0][0])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100*accuracy} % \n Loss: {loss}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Voting)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #R
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixR, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[0][1])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100*accuracyR} % \n Loss: {lossR}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Red)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #G
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixG, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[1][0])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100 * accuracyG} % \n Loss: {lossG}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Green)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    #B
    confusionMatrixPlotAbsolute = sns.heatmap(confusionMatrixB, annot=True,  yticklabels=tickLabels, xticklabels=tickLabels, fmt=".0f", ax=ax[1][1])
    confusionMatrixPlotAbsolute.set_title(f"Accuracy: {100 * accuracyB} %s \n Loss: {lossB}")
    confusionMatrixPlotAbsolute.set_xlabel("Predicted (Blue)")
    confusionMatrixPlotAbsolute.set_ylabel("Actual")

    print(f"LossRGB: {lossRGB}")


    #fig.suptitle(f"Number of Samples {len(testSet)})")

    #determine wrongly classified images (only Vote)
    index = np.invert(equals.numpy().reshape(len(equals)))
    paths = np.array(paths)[index]

    wrongClasses = []
    tpc = top_class.numpy().reshape(len(equals))
    classes = tpc[index]
    
    rightClasses = []
    lbls = labels.view(*top_class.shape).numpy().reshape(len(equals))
    classes1 = lbls[index]

    wrongPredictions = []
    predsAll = prediction.numpy()
    preds = predsAll[index]

    for i in range(len(classes)):
        wrongClasses.append(testSet.classes.get(classes[i]))
        rightClasses.append(testSet.classes.get(classes1[i]))
        tmp = preds[i]
        tmp -= tmp.min()
        tmp /= tmp.sum()

        wrongPredictions.append(tmp)



    return accuracy, confusionMatrix, [paths, wrongClasses, rightClasses, wrongPredictions, index], preds, [labels, ps,psR, psG,psB]
