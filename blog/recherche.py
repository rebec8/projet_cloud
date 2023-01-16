import os
import math
import time
import matplotlib.pyplot as plt
import cv2
import shutil

from distances import extractReqFeatures, getkVoisins

from PIL import Image


def Recherche(filename, descripteurs, features1, test, filenames, distance, top, race, race2):
        start_time = time.time()
        #Remise à 0 de la grille des voisins
        voisins=""
        if descripteurs:
            lvoisins = []
            for key, value in features1.items():
            ##Generer les features de l'images requete
                req = extractReqFeatures(filename, key)

                ##Definition du nombre de voisins
                sortie = int(top)
                #Générer les voisins
                voisins=getkVoisins(req, sortie, value[0], value[1], lvoisins)
            path_image_plus_proches = []
            nom_image_plus_proches = []

            image_path = 'static\\voisins'
            if os.path.isdir(image_path):
                shutil.rmtree(image_path)
            os.mkdir(image_path)
 
            for k in range(sortie*len(descripteurs)):
                path_image_plus_proches.append(voisins[k][0])
                img = os.path.basename(voisins[k][0])
                nom_image_plus_proches.append(img)
                
                if race2 in filenames:
                    shutil.copy(f"{filenames}\\{img}", image_path)

                else:
                    for elt in race:
                        if elt.replace(" ","") in img:
                            shutil.copy(f"{filenames}\\{elt}\\{img}", image_path)

            print(f"tttt : {len(nom_image_plus_proches)}")
                      
        else :
            print("Il faut choisir une méthode !")
        end_time = time.time()
        print(f"Temps mis pour la recherche : {end_time - start_time} secondes")
        return nom_image_plus_proches