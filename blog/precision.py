import time
import matplotlib.pyplot as plt
import os
import shutil

UPLOAD_FOLDER = 'static'

def rappel_precision(filename, sortie, nom_image_plus_proches, animal, race, count):
        start_time = time.time()
        rappel_precision=[]
        rappels=[]
        precisions=[]
        filename_req=os.path.basename(filename)
        num_image, _ = filename_req.rsplit(".")
        num_image = num_image[-4:]
        classe_image_requete = int(num_image)/100
        val =0

        if len(nom_image_plus_proches) == sortie:
            for j in range(sortie):
                classe_image_proche=(int(nom_image_plus_proches[j][-8:-4].split('.')[0]))/100
                classe_image_requete = int(classe_image_requete)
                classe_image_proche = int(classe_image_proche)
                
                if classe_image_requete==classe_image_proche:
                    rappel_precision.append(True) #Bonne classe (pertinant)
                    val += 1
                else:
                    rappel_precision.append(False) #Mauvaise classe (non pertinant)
        else:
            for j in range(sortie):
                classe_image_proche1=(int(nom_image_plus_proches[j][-8:-4].split('.')[0]))/100
                classe_image_proche2=(int(nom_image_plus_proches[j+sortie][-8:-4].split('.')[0]))/100
                classe_image_requete = int(classe_image_requete)
                classe_image_proche1 = int(classe_image_proche1)
                classe_image_proche2 = int(classe_image_proche2)

                if classe_image_requete==classe_image_proche1 or classe_image_requete==classe_image_proche2:
                    rappel_precision.append(True) #Bonne classe (pertinant)
                    val += 1
                else:
                    rappel_precision.append(False) #Mauvaise classe (non pertinant)
        for i in range(sortie): 
            j=i
            val=0
            while(j>=0):
                if rappel_precision[j]:
                    val+=1
                j-=1
            precision = val / (i + 1)
            rappel = val / count
            
            rappels.append(rappel)
            precisions.append(precision)
            #print(f"Précision pour {i+1}/{sortie} : {precision}")
        print(f"Average precision pour {animal}-{race} : {sum(precisions)/len(precisions)}")
        print(f"Val : {val}")
        print(f"Sortie : {sortie}")
        print(f"Len : {count}")
        #Création de la courbe R/P
        fig = plt.figure()
        plt.plot(rappels,precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{animal}-{race} : R/P {str(sortie)} voisins de l'image n°{num_image}")

        #Enregistrement de la courbe RP

        image_path = UPLOAD_FOLDER

        save_folder=os.path.join(".",num_image)
        if not os.path.exists(f"{image_path}/{save_folder}"):
            os.makedirs(f"{image_path}\{save_folder}")

        save_image = os.path.join(f"{save_folder}",num_image+'.png')
        save_image = save_image.replace("\\", "/")
       
        save_name=os.path.join(f"{image_path}/{save_folder}",num_image+'.png')
        plt.savefig(save_name,format='png',dpi=600)
        plt.close()
        
        end_time = time.time()
        print(f"Temps mis pour la courbe R/P : {end_time - start_time} secondes")
        return save_image.lstrip('.'), precisions[-1], rappels[-1], sum(precisions)/len(precisions)