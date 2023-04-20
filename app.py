from flask import Flask, request, send_file
import flask

import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import asarray

import cv2
import tensorflow as tf

from PIL import Image
from werkzeug.utils import secure_filename
import os, shutil, io, secrets, string
from imageio import imread
import sqlite3
from mtcnn.mtcnn import MTCNN
import base64

from db import db_init, db
from dbModels import Img

from keras.preprocessing.text import Tokenizer, one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Embedding, LSTM, Dropout, BatchNormalization, Dense, concatenate, Flatten, Conv1D
from keras.optimizers import RMSprop, Adam

# from keras_vggface.vggface import VGGFace
from glob import glob
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Lambda, Reshape
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from keras_vggface.utils import preprocess_input

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db_init(app)
    return app 

APP_ROOT = os.path.dirname(os.path.abspath('__file__'))

model = load_model('vgg_face.h5')

#------------------------
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img):
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    #print("found faces: ", len(faces))

    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return (True, img)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (False, None)
        #raise ValueError("No face found in the passed image ")

def alignFace(img_decode):
    img = img_decode
#     plt.imshow(img[:, :, ::-1])
#     plt.show()

    img_raw = img.copy()

    img, gray_img = detectFace(img)
    
    eyes = eye_detector.detectMultiScale(gray_img)
    
    #print("found eyes: ",len(eyes))
    
    if len(eyes) >= 2:
        #find the largest 2 eye
        
        base_eyes = eyes[:, 2]
        #print(base_eyes)
        
        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)
        
        df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
        
        eyes = eyes[df.idx.values[0:2]]
        
        #--------------------
        #decide left and right eye
        
        eye_1 = eyes[0]; eye_2 = eyes[1]
        
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        #--------------------
        #center of eyes
        
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        #center_of_eyes = (int((left_eye_x+right_eye_x)/2), int((left_eye_y+right_eye_y)/2))
        
        cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
        #cv2.circle(img, center_of_eyes, 2, (255, 0, 0) , 2)
        
        #----------------------
        #find rotation direction
        
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock
            print("rotate to inverse clock direction")
        
        #----------------------
        
        cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
        
        cv2.line(img,right_eye_center, left_eye_center,(67,67,67),1)
        cv2.line(img,left_eye_center, point_3rd,(67,67,67),1)
        cv2.line(img,right_eye_center, point_3rd,(67,67,67),1)
        
        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)
        
        #print("left eye: ", left_eye_center)
        #print("right eye: ", right_eye_center)
        #print("additional point: ", point_3rd)
        #print("triangle lengths: ",a, b, c)
        
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        #print("cos(a) = ", cos_a)
        angle = np.arccos(cos_a)
        #print("angle: ", angle," in radian")
        
        angle = (angle * 180) / math.pi
        print("angle: ", angle," in degree")
        
        if direction == -1:
            angle = 90 - angle
        
        print("angle: ", angle," in degree")
        
        #--------------------
        #rotate image
        
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))
    
    return (True, new_img)
    
#------------------------

#opencv path

opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]

path = folders[0]
for folder in folders[1:]:
    path = path + "/" + folder

face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
eye_detector_path = path+"/data/haarcascade_eye.xml"
nose_detector_path = path+"/data/haarcascade_mcs_nose.xml"

if os.path.isfile(face_detector_path) != True:
    raise ValueError("Confirm that opencv is installed on your environment! Expected path violated.")

face_detector = cv2.CascadeClassifier(face_detector_path)
eye_detector = cv2.CascadeClassifier(eye_detector_path) 
nose_detector = cv2.CascadeClassifier(nose_detector_path) 

#------------------------

# test_set = ["../input/offspring/C_0010.jpg"]

# for instance in test_set:
#     alignedFace = alignFace(instance)
#     plt.imshow(alignedFace[:, :, ::-1])
#     plt.show()
    
#     img, gray_img = detectFace(alignedFace)
#     plt.imshow(img[:, :, ::-1])
#     plt.show()

def align_crop_resize(fnames, height=None, width= None): 
#     aligned_dir=os.path.join(dest_dir, 'Aligned Images')
#     if os.path.isdir(dest_dir):
#         shutil.rmtree(dest_dir)
#     os.mkdir(dest_dir)  #start with an empty destination directory
#     os.mkdir(aligned_dir)
#     os.mkdir(dest_dir)
#     print(cropped_dir)
    success=[]
    for fname in fnames:
        try:
            img = Img.query.filter_by(name=fname).first()

            # Convert the byte string to a numpy array
            nparr = np.fromstring(img.img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            shape=img.shape
            status,img=alignFace(img) # rotates the image for the eyes are horizontal
            print("Status: ", status)
            if status:
    #                     aligned_path=os.path.join(aligned_dir,f)
    #                     cv2.imwrite(aligned_path, img)                    
                cstatus, img=detectFace(img) # crops the aligned image to return the largest face
    #                     plt.imshow(img[:, :, ::-1])
    #                     plt.show()
                print("Cstatus: ", cstatus)
                if cstatus:
                    if height != None and width !=None:
                        img=cv2.resize(img, (height, width)) # if height annd width are specified resize the image
                    image_bytes = bytes(cv2.imencode('.jpg', img)[1])
                    imgFile = Img(img=image_bytes, name=changeFilename(fname), mimetype='image/jpeg')
                    db.session.add(imgFile)
                    db.session.commit()
                    success.append(removeHashname(fname)) # update the count of successful processed images
        except:
            print('file', removeHashname(fname), 'is a bad image file')
    return success

def read_img(fname):
    img = Img.query.filter_by(name=fname).first()
    img = Image.open(io.BytesIO(img.img))
    img = img.resize((197, 197))
    # img = tf.keras.utils.load_img(path, target_size=(197, 197))
    img = np.array(img).astype(np.float64)
    return preprocess_input(img, version=2)

def changeFilename(fname):
    filename, file_extension = os.path.splitext(fname)
    fnameChanged = filename + '-DataCleaned' + file_extension
    return fnameChanged

def removeHashname(fname):
    filename, file_extension = os.path.splitext(fname)
    hashRemoved = filename.rsplit("-")[0:-1]
    fname = '-'.join(hashRemoved)
    fnameChanged = fname + file_extension
    return fnameChanged

@create_app.app.route('/')
@create_app.app.route('/home')
def upload_image():
    return flask.render_template('index.html')


@create_app.app.route('/upload', methods=['POST']) #POST will get the data and perform operatins
# def findNgrokUrl():
#     ngrok_tunnel1 = ngrok.connect()
#     return ngrok_tunnel1

def post_image():
    global graph

    if flask.request.method == 'POST':

        pic1 = request.files['file1']
        if not pic1:
            return 'No pic uploaded!', 400

        filename = secure_filename(pic1.filename)
        mimetype = pic1.mimetype
        if not filename or not mimetype:
            return 'Bad upload!', 400

        filename, file_extension = os.path.splitext(filename)
        fname1 = filename + '-' +  ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(15)) + file_extension
        img1 = Img(img=pic1.read(), name=fname1, mimetype=mimetype)
        db.session.add(img1)
        db.session.commit()

        pic2 = request.files['file2']
        if not pic2:
            return 'No pic uploaded!', 400

        filename = secure_filename(pic2.filename)
        mimetype = pic2.mimetype
        if not filename or not mimetype:
            return 'Bad upload!', 400

        filename, file_extension = os.path.splitext(filename)
        fname2 = filename + '-' +  ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(15)) + file_extension
        img2 = Img(img=pic2.read(), name=fname2, mimetype=mimetype)
        db.session.add(img2)
        db.session.commit()

        pic3 = request.files['file3']
        if not pic3:
            return 'No pic uploaded!', 400

        filename = secure_filename(pic3.filename)
        mimetype = pic3.mimetype
        if not filename or not mimetype:
            return 'Bad upload!', 400

        filename, file_extension = os.path.splitext(filename)
        fname3 = filename + '-' +  ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(15)) + file_extension
        img3 = Img(img=pic3.read(), name=fname3, mimetype=mimetype)
        db.session.add(img3)
        db.session.commit()

        fnames = [fname1, fname2, fname3]
        fnamesUncropped = [removeHashname(fname1), removeHashname(fname2), removeHashname(fname3)]

        success = align_crop_resize(fnames)

        unsuccessfulCropped = [i for i in fnamesUncropped if i not in success]
        if len(unsuccessfulCropped) == 1:
            message = "{} is a bad image file. Face can't be extracted, please upload a different image.".format(unsuccessfulCropped)
            return flask.render_template('error.html', message = message)
        elif len(unsuccessfulCropped) > 1:
            message = "{} are a bad image files. Faces can't be extracted, please upload a different image.".format(unsuccessfulCropped)
            return flask.render_template('error.html', message = message)

        else:
            predictions = []

            X1 = np.array([read_img(changeFilename(fname1))])
            X3 = np.array([read_img(changeFilename(fname3))])

            pred = model.predict([X1, X3]).ravel().tolist()
            predictions += pred

            X2 = np.array([read_img(changeFilename(fname2))])
            X3 = np.array([read_img(changeFilename(fname3))])

            pred = model.predict([X2, X3]).ravel().tolist()
            predictions += pred

            predictions = [i*100 for i in predictions]
            print("Predictions: ", predictions)

            return flask.render_template('end.html', image_name1=changeFilename(fname1), image_name2=changeFilename(fname2), image_name3=changeFilename(fname3), pred1=predictions[0], pred2=predictions[1])
    
@create_app.app.route('/i/<ident>')
def profile_image(ident):

    img = Img.query.filter_by(name=ident).first()
    if not img:
        return 'Img Not Found!', 404
    
    return send_file(io.BytesIO(img.img), mimetype=img.mimetype)  

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    create_app.app.run()