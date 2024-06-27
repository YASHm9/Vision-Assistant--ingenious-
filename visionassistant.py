from __future__ import division, print_function

import datetime
from yolov5 import detect
import speech_recognition as sr
import pytesseract
import pyttsx3
import time
import vgg16_places_365 as vgg16_places_365
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# -*- coding: utf-8 -*-
'''VGG16-places365 model for Keras

# Reference:
- [Places: A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf)
'''
import os
import time
import numpy as np
from cv2 import resize
import cv2
import openai
from wrapt_timeout_decorator import *

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

obj=["detect the objects","the objects", "the object", "detect object","detect objects","objects near me","objects near","objects"]
txt1 = ["read the text","read for me","text","texts","read the texts","read the document","read the documents","read document"]
env1= ["where am i","where i am","scenario","detect the environment","detect the environments"]
emtn = ["detect emotion","detect the emotion","detect the emotions","detect emotions"]

@timeout(30)
def objdetect():
        detect.run(weights= "yolov5m.pt",source = 0)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def textdetection():
    

    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    counter = 0
    start_time = time.time()
    while True:

        ret, frame = cap.read()
 
        cv2.imshow("Webcam", frame)
 
        if counter % 60 == 0:
            cv2.imwrite("test1.jpg", frame)
            print("Image saved!")
        counter += 1
        if time.time()-start_time>5:
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    img = cv2.imread("test1.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))


    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)


    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
												cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()




    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        speak(text)

def scenariodetection():
    WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if __name__ == '__main__':
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        counter = 0
        start_time = time.time()
        while True:

            ret, frame = cap.read()
        # Display the frame
            cv2.imshow("Webcam", frame)
        # Auto save image every 10 seconds
            if counter % 60 == 0:
                cv2.imwrite("test1.jpg", frame)
                print("Image saved!")
            counter += 1
            if time.time()-start_time>5: # Stop the loop after 30 sec
                break
    # Check if the user pressed 'q' to exit
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

# Release the webcam and close the display window
        cap.release()
        cv2.destroyAllWindows()

        image = cv2.imread("test1.jpg")
        image = np.array(image, dtype=np.uint8)
        image = resize(image, (224, 224))
        image = np.expand_dims(image, 0)

        model = vgg16_places_365.VGG16_Places365(weights='places')
        predictions_to_return = 5
        preds = model.predict(image)[0]
        top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

    # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        print('--PREDICTED SCENE CATEGORY:')
    # output the prediction
        for i in range(0, 2):
            print(classes[top_preds[i]])
            speak(classes[top_preds[i]])

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.adjust_for_ambient_noise(source,duration=1)
        r.pause_threshold = 1
        audio = r.listen(source)
    try: 
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        print(f"user said: {query}\n")
    except Exception as e:
        print('Say that again please...')
        speak("say that again please")
        return "None"
    return query

@timeout(20)
def emotiondetect():
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

    classifier =load_model(r'model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)



    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                speak(label)
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('Emotion Detector',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak('Good Morning!')
    elif hour>=12 and hour<18:
        speak('Good Afternoon!')
    else:
        speak('Good Evening')
    speak('I am your virtual Assistant. please tell me how may i help you')

def gpt3(texts):

    openai.api_key = "lorem ipsum"

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt = f"Iam a assistant for visually impaired, i was built to help them by comunicating with them \nYou: What have you been up to?\nassistant: nothing just waiting to speak with you.\nYou: hi bro assistant: hello, how can i help you?  \nYou: {texts}\nassistant:",
    temperature=0.7,
    max_tokens=256, 
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    content = response.choices[0].text.split('.')
    return response.choices[0].text

if __name__ == '__main__':
    wishMe()
    while True:
        query = takecommand().lower()
        if "bro" in query:
            response = gpt3(query)
            speak(response)
            print(response)
        for i in obj:

            if i in query:
                try:
                    objdetect()
                except:
                    pass
        else:
            for i in txt1:
                if i in query:
                    textdetection()
            for i in env1:
                if i in query:
                    scenariodetection()
            for i in emtn:
                if i in query:
                    try:
                     emotiondetect()
                    except:
                       pass
       
