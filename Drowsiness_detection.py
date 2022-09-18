import argparse
import numpy as np
import cv2 as cv
from scipy.spatial import distance
from imutils.video import VideoStream
from imutils import face_utils
import threading as Thread
import imutils
import time
import playsound
import pygame
import argparse
import dlib

# Scipy is used to calculate the Euclediam Distance Between the facial landMarks point in the Eye Espect Ratio
# imutils ,my image of computer vision and image processing functions to make working with the OpenCV easier

def sound_alarm(path):
    pygame.init()
    pygame.mixer.music.load(path)   # enter the Path of alarm "alarm.wav"
    pygame.mixer.music.play()
    time.sleep(2)
    pygame.mixer.music.stop()
    # playsound.playsound(path)

def eye_aspect_ratio(eye):
    # computing the computing distance between the two set of vertical eye landmrks (x,y) coordinates
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])

    #computing the euclidean diatance b/w horizontal eye landmarks 
    C=distance.euclidean(eye[0],eye[3])
    #Eye Aspect ratio
    EAR=(A+B)/(2.0*C)
    return EAR
#The return value of EAR will be approximately constant when th eye is open.The value will rapidly decrase when toward Zero during the blink
# if the Eye is closed ,the EAR will remain constant approximately,but will be much smalller than the ratio when eye is open

# construct the argument parse and parse arguments
ap =argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
ap.add_argument("-a","--alarm",type=str,default="",help="path alarm .WAV file")
ap.add_argument("-w","--webcam",type=int ,default=0,help="index of webcam on system")

args =vars(ap.parse_args())


# Our drowsiness detector requires one command line argument followed by two optional ones

#--shape predictor: this the path to dlib pre trainde facal land mark detector

#--alarm: With the help of pygame library files

# defined two constants, one for the eye aspect ratio to indicate the blink of the eye
# second constant  for the number of consecutive frames the eye must be below the threshold for the set off the alarm

EYE_AR_THRESH=0.3    # if the EAR falls below this threshold ,we will start conunting the the number of times the person has close their eyes

EYE_AR_CONSCE_FRAMES=48 # if the number of the frames the person has closed their eyes in exceeds We will sound the alarm

#initialize the frame counter as well as boolean used
# indicate if the alarm is going off

#define the Counter ,the total number of consecutive frames where the eye aspect ratio is below EYE_AR_THRESH
# if the Counter exceed the EYE_ARCONSEC_THRESH,then we will update the boolean ALARM_ON
Counter=0
ALARM_ON=False


# Facial land Mark Detector Using dlib files in python

# initilaizing the dlib face detetctor and the create the facial land mark predictor

print("Information Loading facial landmark predictor....")

detector=dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor(args["shape_predictor"])


# to extract the eye regions from the set of facial landmarks
# extracting the indexs of the facial landmarks for the left and the right eye,respectively
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# starting the core of our drowsiness detector

# starting the video Stream
print("Startin the video Stream thrwamd....")

vs = VideoStream(src=args["webcam"]).start()  #capturing the frame by frame from the webcam
time.sleep(1.0)

#loop over frames from the video stream
while True:
    # taking frame by frame in threaded video file stream
    # converting into the fray scale channel
    frame=vs.read()
    # reading the next frame ,which we when the preprocess by resizing it to have a witdh 0 450 pixels and conveting it to the grayScale mode
    frame=imutils.resize(frame,width=450)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # detect the faces in the grayScale frame
    rects = detector(gray,0)

    # loop for face detetction
    for rect in rects:
        # determin the facial landmarks for the fface region,then
        # convert the facial land marks as (x,y) coordinates to a numpy array
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        # extarcting the left and right eye coordinate 
        # and the coordinates  to compute the eye EAR for both the eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR =  eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average of the EAR together for both the eye
        ear = (leftEAR+rightEAR)/2.0

        # drawing the converHull for the left eye and the right eye
        # Visualizing each eye
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)

        cv.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv.drawContours(frame,[rightEyeHull],-1,(0,255,0),-1)

        # checking if the EAR ratio in below the blink threshold
        # and if so,incrementing the blinking counter
        if ear<EYE_AR_THRESH:
            Counter +=1

            #if the eyes are closed for the sufficient nuber of the time
            # the we should sound the alarm

            if Counter>=EYE_AR_CONSCE_FRAMES:
                # if the alarm is not on then turn the alarm
                if not ALARM_ON:
                    ALARM_ON =True

                    # Checking to see if the alarm is supplied or not
                    # and if so,start the thread to have the alarm
                    # sound will be played in the background
                    if args["alarm"]!="":
                        t= Thread.Thread(target=sound_alarm, args=(args["alarm"],))
                        t.deamon=True
                        t.start()

                    # draw an alarm to the frame
                    cv.putText(frame,"Drowsiness Alert!",(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                else:
                    Counter=0
                    ALARM_ON=False

            # Drawing the EAR Value on the frame
            cv.putText(frame,"EAR:{:.2f}".format(ear),(300,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            # show the frame
    cv.imshow("Frame",frame)
    key=cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
vs.stop()







