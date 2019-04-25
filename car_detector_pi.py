from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
#from threading import Thread

import numpy as np

import imutils
import time
import cv2
#import os

# Define path to Car/Not Car model
MODEL_PATH = "/home/pi/Car_Detector_Pi/car_model_pi.h5"

# Initialize the total number of frames that consecutively contain car
TOTAL_CONSEC = 0
TOTAL_THRESH = 20

# Car alarm has been triggered
#CAR = False

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)

# Initialize the video stream and allow the camera sensor to warm up
print("Starting video stream...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    # Prepare the image to be classified by deep learning network
    image = cv2.resize(frame, (100, 100))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    # Classify the input image and initialize the label and
    # probability of the prediction
    """(notCar, car) = model.predict(image)[0]
    label = "Not Car"
    proba = notCar

    # Check to see if car was detected using CNN
    if car > notCar:
        # Update the label and prediction probability
        label = "Car"
        proba = car

        # Increment the total number of consecutive frames that contain car
        #TOTAL_CONSEC += 1"""
    car = model.predict(image)[0]
    label = "Not Car"
    proba = float(car)
    
    if proba > .75:
        label = "Car"

    # Build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
print("Cleaning up...")
cv2.destroyAllWindows()
vs.stop()
