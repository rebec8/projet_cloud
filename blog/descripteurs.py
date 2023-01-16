import os
import numpy as np
import shutil
import cv2
import skimage
import itertools
import warnings
warnings.filterwarnings("ignore")

from skimage import data, img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

    
def generateBGR(filenames, animal, race, i):
    desc = "BGR"

    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

        num_image, _ = path.split(".")
        #if animal in num_image and race in num_image:
        np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,feature)
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateHSV(filenames, animal, race, i):
    desc = "HSV"
   
    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")


    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
        histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
        histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        num_image, _ = path.split(".")
        np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,feature)
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateSIFT(filenames, animal, race, i):
    desc = "SIFT"


    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

    
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        featureSum = 0
        sift = cv2.SIFT_create()  
        kps , des = sift.detectAndCompute(img,None)
   
        num_image, _ = path.split(".")
        if len(kps) > 0:
          np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,des)
        featureSum += len(kps)
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateORB(filenames, animal, race, i):
    desc = "ORB"

    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

   
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        orb = cv2.ORB_create()
        key_point1,descrip1 = orb.detectAndCompute(img,None)
        
        num_image, _ = path.split(".")
        if len(key_point1) > 0:
          np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,descrip1 )
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateLBP(filenames, animal, race, i):
    desc = "LBP"
    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

    
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        points=8
        radius=1
        method='default'
        subSize=(70,70)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(350,350))
        fullLBPmatrix = local_binary_pattern(img,points,radius,method)
        histograms = []
        for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                histograms = np.concatenate((histograms,subHist),axis=None)
        num_image, _ = path.split(".")
        np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,histograms)
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateGLCM(filenames, animal, race, i):
    desc = "GLCM"
    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

    
    distances=[1,-1]
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
    for path in os.listdir(filenames):
        image = cv2.imread(filenames+"/"+path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcmMatrix = skimage.feature.graycomatrix(gray, distances=distances, angles=angles, normed=True)
        glcmProperties1 = skimage.feature.graycoprops(glcmMatrix,'contrast').ravel()
        glcmProperties2 = skimage.feature.graycoprops(glcmMatrix,'dissimilarity').ravel()
        glcmProperties3 = skimage.feature.graycoprops(glcmMatrix,'homogeneity').ravel()
        glcmProperties4 = skimage.feature.graycoprops(glcmMatrix,'energy').ravel()
        glcmProperties5 = skimage.feature.graycoprops(glcmMatrix,'correlation').ravel()
        glcmProperties6 = skimage.feature.graycoprops(glcmMatrix,'ASM').ravel()
        feature = np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel()
        num_image, _ = path.split(".")
        np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,feature)
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def generateHOG(filenames, animal, race, i):
    desc = "HOG"
    

    if not os.path.isdir(f"{desc}"):
        os.mkdir(f"{desc}")
    if not os.path.isdir(f"{desc}/{animal}"):
        os.mkdir(f"{desc}/{animal}")
    if i == 0 and os.path.isdir(f"{desc}/{animal}"):
        shutil.rmtree(f"{desc}/{animal}")
        os.mkdir(f"{desc}/{animal}")

   
    cellSize = (25,25)
    blockSize = (50,50)
    blockStride = (25,25)
    nBins = 9
    winSize = (350,350)
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,winSize)
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
        feature = hog.compute(image)
        num_image, _ = path.split(".")
        np.savetxt(f"{desc}/{animal}/{str(num_image)}.txt" ,feature )
        if race.replace(" ","") in path:
            i+=1
    return f"{desc}/{animal}",i


def combiner(descripteurs,filenames,animal,race,count):
    combined_Features=[]
    for descripteur in descripteurs:
        count = 0
        if descripteur == "BGR":
            folder, count = generateBGR(filenames, animal, race, count)
        elif descripteur == "HSV":
            folder, count = generateHSV(filenames, animal, race, count)
        elif descripteur == "SIFT":
            folder, count = generateSIFT(filenames, animal, race, count)
        elif descripteur == "ORB":
            folder, count = generateORB(filenames, animal, race, count)
        elif descripteur == "GLCM":
            folder, count = generateGLCM(filenames, animal, race, count)
        elif descripteur == "LBP":
            folder, count = generateLBP(filenames, animal, race, count)
        elif descripteur == "HOG":
            folder, count = generateHOG(filenames, animal, race, count)

        combined_Features.append(folder)
    
    return combined_Features, count