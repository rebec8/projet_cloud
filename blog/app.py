from flask import Flask,render_template, redirect, request, flash, url_for, session
import urllib.request
import os
import json
import shutil
import time
#from tensorflow.keras.preprocessing.image import load_img
from werkzeug.utils import secure_filename

from descripteurs import* #generateBGR, generateGLCM, generateHOG, generateHSV, generateLBP, generateORB, generateSIFT
from distances import loadfeatures
from recherche import Recherche
from precision import rappel_precision

app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "my key"
app.config['IMAGE_UPLOADS'] = 'C:/Users/fotso/OneDrive/Documents/MA2/Q1/Multimedia Retrieval and cloud computing/projet_cloud/projet_MIR_final2/blog/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def blog():
    return render_template('blog/index.html')


@app.route('/', methods=["POST","GET"])
def upload_image():
    if request.method == "POST":
        
        if request.form["action"] == "Charger descripteurs sous-classe":
            niveau = 2
            descripteurs= request.form.getlist("horns")

            session['descripteurs'] = descripteurs

            filename = session.get("filename")

            if "chiens" in filename:
                animal = "chiens"
                if "golden" in filename:
                    race2 = "golden retriever"
                elif "boxer" in filename:
                    race2 = "boxer"
                else:
                    race2 = "Rottweiler"

            elif "poissons" in filename:
                animal = "poissons"
                if "eagle" in filename:
                    race2 = "eagle ray"
                elif "hammerhead" in filename:
                    race2 = "hammerhead"
                else:
                    race2 = "tiger shark"
            else:
                animal = "singes"
                if "chimpanzee" in filename:
                    race2 = "chimpanzee"
                elif "gorilla" in filename:
                    race2 = "gorilla"
                else:
                    race2 = "squirrel monkey"


            filenames = f"MIR_DATASETS_B\\{animal}\\{race2}"
            session["path"] = filenames
            count = 0

            if len(descripteurs) == 1:
                start_time = time.time()
                if descripteurs[0] == "BGR":
                    fold, count = generateBGR(filenames, animal, race2, count)
                elif descripteurs[0] == "HSV":
                    fold, count = generateHSV(filenames, animal, race2, count)
                elif descripteurs[0] == "SIFT":
                    fold, count = generateSIFT(filenames, animal, race2, count)
                elif descripteurs[0] == "ORB":
                    fold, count = generateORB(filenames, animal, race2, count)
                elif descripteurs[0] == "GLCM":
                    fold, count = generateGLCM(filenames, animal, race2, count)
                elif descripteurs[0] == "LBP":
                    fold, count = generateLBP(filenames, animal, race2, count)
                elif descripteurs[0] == "HOG":
                    fold, count = generateHOG(filenames, animal, race2, count)
                folder = []
                folder.append(fold)
                end_time = time.time()
            else:
                start_time = time.time()
                folder, count=combiner(descripteurs,filenames,animal,race2,count)
                end_time = time.time()

            print(f"Temps d'indexation : {end_time - start_time}")

            session["folder"] = folder
            session["animal"] = animal
            session["race2"] = race2
            session['count'] = count
            session['niveau'] = niveau
            
            if filename:
                return render_template("blog/index.html", filename=filename, niveau=niveau,descripteurs=descripteurs)
            else:
                return render_template("blog/index.html", descripteurs=descripteurs)


        if request.form["action"] == "Charger descripteurs classe":
            niveau = 1
            descripteurs= request.form.getlist("horns")
            session['descripteurs'] = descripteurs

            filename = session.get("filename")

            if "chiens" in filename:
                animal = "chiens"
                if "golden" in filename:
                    race2 = "golden retriever"
                elif "boxer" in filename:
                    race2 = "boxer"
                else:
                    race2 = "Rottweiler"

            elif "poissons" in filename:
                animal = "poissons"
                if "eagle" in filename:
                    race2 = "eagle ray"
                elif "hammerhead" in filename:
                    race2 = "hammerhead"
                else:
                    race2 = "tiger shark"
            else:
                animal = "singes"
                if "chimpanzee" in filename:
                    race2 = "chimpanzee"
                elif "gorilla" in filename:
                    race2 = "gorilla"
                else:
                    race2 = "squirrel monkey"
                
            path = f"MIR_DATASETS_B\\{animal}"
            session['path'] = path

            race = []

            count = 0
            
            if len(descripteurs) == 1:
                for i in os.listdir(path):
                    filenames = f"{path}\\{i}"
                    if descripteurs[0] == "BGR":
                        fold, count = generateBGR(filenames, animal, race2, count)
                    elif descripteurs[0] == "HSV":
                        fold, count = generateHSV(filenames, animal, race2, count)
                    elif descripteurs[0] == "SIFT":
                        fold, count = generateSIFT(filenames, animal, race2, count)
                    elif descripteurs[0] == "ORB": 
                        fold, count = generateORB(filenames, animal, race2, count)
                    elif descripteurs[0] == "GLCM":
                        fold, count = generateGLCM(filenames, animal, race2, count)
                    elif descripteurs[0] == "LBP":
                        fold, count = generateLBP(filenames, animal, race2, count)
                    elif descripteurs[0] == "HOG":
                        fold, count = generateHOG(filenames, animal, race2, count)
                    folder = []
                    folder.append(fold)
                    race.append(i)
            else:
                folder, count=combiner(descripteurs, filenames, animal, race, count)

            
            session["folder"] = folder
            session["animal"] = animal
            session["race"] = race
            session["race2"] = race2
            session['count'] = count

            
            
            if filename:
                return render_template("blog/index.html", filename=filename, descripteurs=descripteurs, niveau=niveau)
            else:
                return render_template("blog/index.html", descripteurs=descripteurs)
        if request.form["action"] == "Recherche":

            distance = request.form.getlist("distances")
            top = request.form["top"]

            filename = session.get("filename")
            descripteurs = session.get("descripteurs")
            folder = session.get("folder")
            animal = session.get("animal")
            race = session.get("race")
            race2 = session.get("race2")
            niveau = session.get("niveau")

            path = session.get("path")
            
            #for i in os.listdir(path):
            filenames = path
            features = {}
            
            for i in range(len(descripteurs)):
                print('OK')
                features1 = []
                features[descripteurs[i]] = []
                features[descripteurs[i]].append(distance[i])
                features[descripteurs[i]].append(loadfeatures(folder[i], filenames, features1))

            features1 = []
            for i in range(len(descripteurs)):
                test = loadfeatures(folder[i], filenames, features1)
                

            nom_image_plus_proches = Recherche(filename, descripteurs, features, test, filenames, distance, top, race, race2)
            session["nom_image_plus_proches"] = nom_image_plus_proches
            session["top"] = top

            
            return render_template("blog/index.html", filename=filename, descripteurs=descripteurs, nom_image_plus_proches=nom_image_plus_proches)

    
        if request.form["action"] == "Calculer la courbe R/P":
            filename = session.get("filename")
            descripteurs = session.get("descripteurs")
            nom_image_plus_proches = session.get("nom_image_plus_proches")
            animal = session.get("animal")
            race2 = session.get("race2")
            top = session.get("top")
            count = session.get("count")
            niveau = session.get("niveau")

            save_image, prec, rap, ave_prec = rappel_precision(filename, int(top), nom_image_plus_proches, animal, race2, count)

            print(f"nnn : {niveau}")
            return render_template("blog/index.html", filename=filename, descripteurs=descripteurs, nom_image_plus_proches=nom_image_plus_proches, save_image=save_image, prec=prec, rap=rap, ave_prec=ave_prec, niveau=niveau)

        else:
            session.clear()
            print(request.files)
            image = request.files['file']


            if image.filename == "":
                print("File name is invalid")
                return redirect(request.url)

            filename = secure_filename(image.filename)
            if os.path.isdir(UPLOAD_FOLDER):
                shutil.rmtree(UPLOAD_FOLDER)
            os.mkdir(UPLOAD_FOLDER)
            basedir = os.path.abspath(os.path.dirname(__file__))
            image.save(os.path.join(basedir, app.config["IMAGE_UPLOADS"],filename))
            
            print(f"Image {filename} bien upload")
            session['filename'] = filename
            return render_template("blog/index.html", filename=filename)
        
    return render_template("blog/index.html")


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename = '/uploads/'+filename), code=301)        
"""
@app.route('/', methods=['GET','POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("Pas d'image sélectionné")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('Upload filename:' + filename)
        flash('Image upload et affichée avec succès')
        return render_template('index.html', filename=filename)
    else:
        flash("Format d'image autorisé : png, jpg, jpeg, gif")
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
"""
@app.errorhandler(404)
def page_not_found(error):
    return render_template('errors/404.html'), 404

if __name__== '__main__':
    app.run(debug=True) #app.run(debug=True,port=3000)