import math
import numpy as np
import time
import cv2
import os
import operator

import skimage

from skimage import data, img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize



def euclidianDistance(l1,l2):
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += np.square(l1[i]-l2[i])
    return math.sqrt(distance)
  
  
def euclidean(l1, l2):
    dist = np.sqrt(np.sum(np.square(l1-l2)))
    return dist


def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1,l2,dtype=np.float64)),dtype=np.float64)
    den = np.sqrt(np.sum(l1,dtype=np.float64)*np.sum(l2,dtype=np.float64))
    return math.sqrt( 1 - num / den )


def chiSquareDistance(l1, l2):
    s = 0.0
    for i,j in zip(l1,l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
    return s


def flann(a,b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)


def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)


def getkVoisins(test, k, distance, features, lvoisins) :
    ldistances = []
    for i in range(len(features)):
        dist = distance_f(test, features[i][1], distance)
        ldistances.append((features[i][0], features[i][1], dist))
    ldistances.sort(key=operator.itemgetter(2))
    for i in range(k):
        lvoisins.append(ldistances[i])
    print(f"taille lvoisins : {len(lvoisins)}")
    return lvoisins


def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = euclidean(l1,l2)
    elif distanceName in ["Correlation","Chicarre","Intersection","Bhattacharyya"]:
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chicarre":
            #distance = chiSquareDistance(l1,l2)
            methode=cv2.HISTCMP_CHISQR
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Bhattacharyya":
            distance = bhatta(l1,l2)  
    elif distanceName=="Brute force":
        distance = bruteForceMatching(l1,l2)
    elif distanceName=="Flann":
        distance = flann(l1,l2)
    return distance


def loadfeatures(folder_model, filenames, features1):
    ##Charger les features de la base de données.
        features_copy = []
        pas=0
        print("chargement de descripteurs en cours ...")
        for j in os.listdir(folder_model): #folder_model : dossier de features
            data=os.path.join(folder_model,j)
            if not data.endswith(".txt"):
                continue
            feature = np.loadtxt(data)
            features_copy.append((os.path.join(filenames,os.path.basename(data).split('.')[0]+'.jpg'),feature))
            pas += 1
        if len(features1) > 0:
            features = np.concatenate((features1, features_copy), axis=None)
            return features
        else:
            features1 = features_copy.copy()
            return features1


def extractReqFeatures(fileName, key):  
    compt = 0
    if fileName : 
        img = cv2.imread(f"static/uploads/{fileName}")
        if key=="BGR": #Couleurs
            compt += 1
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif key=="HSV": # Histo HSV
            compt += 1
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif key=="SIFT": #SIFT
            compt += 1
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif key=="ORB": #ORB
            compt += 1
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)

        elif key=="GLCM": #GLCM
            compt += 1
            distances=[1,-1]
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = img_as_ubyte(gray)
            glcmMatrix = skimage.feature.greycomatrix(gray, distances=distances, angles=angles, normed=True)
            glcmProperties1 = skimage.feature.greycoprops(glcmMatrix,'contrast').ravel()
            glcmProperties2 = skimage.feature.greycoprops(glcmMatrix,'dissimilarity').ravel()
            glcmProperties3 = skimage.feature.greycoprops(glcmMatrix,'homogeneity').ravel()
            glcmProperties4 = skimage.feature.greycoprops(glcmMatrix,'energy').ravel()
            glcmProperties5 = skimage.feature.greycoprops(glcmMatrix,'correlation').ravel()
            glcmProperties6 = skimage.feature.greycoprops(glcmMatrix,'ASM').ravel()
            vect_features = np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel()
    
        elif key=="LBP": #LBP
            compt += 1
            points=8
            radius=1
            method='default'
            subSize=(70,70)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(350,350))
            fullLBPmatrix = local_binary_pattern(img,points,radius,method)
            vect_features = []
            for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
                for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                    subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                    subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                    vect_features = np.concatenate((vect_features,subHist),axis=None)

        elif key=="HOG": #HOG 
            compt += 1
            cellSize = (25,25)
            blockSize = (50,50)
            blockStride = (25,25)
            nBins = 9
            winSize = (350,350)
            
            image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,winSize)
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
            vect_features = hog.compute(image)

			
        #np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        #print("saved")
        #print("vect_features", vect_features)
        print(type(vect_features))
        return vect_features