# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:01:25 2022

@author: isis.bonet
"""

#conda install -c menpo opencv
#pip install opencv-python

import cv2 # opencv

########Probar cámara
video_capture = cv2.VideoCapture(0)
while(True):
    ret, frame = video_capture.read()
    cv2.imshow('frame', frame)  # Mostrar el frame capturado con la cámara
    if cv2.waitKey(1) & 0xFF == 27: #detener con tecla Esc
        break
video_capture.release()
cv2.destroyAllWindows()













#Probar AWS

##################################################################
def openImage(file_image):
  with open(file_image, 'rb') as openfile:
    content = openfile.read()
  return content

import matplotlib.pyplot as plt
import boto3
import io
from PIL import Image
import os

import configparser
config = configparser.ConfigParser()
config.read('C:/Users/isis.bonet/Downloads/credencials.txt')

###Comprobar conexión con AWS
path_image='C:/Users/isis.bonet/Downloads/'
client = boto3.client(service_name='rekognition', region_name="us-east-2",
                      aws_access_key_id=config.get('default', 'aws_access_key_id'), 
                      aws_secret_access_key=config.get('default', 'aws_secret_access_key')
                    )

image=openImage(path_image+'face.jpg')
response = client.detect_faces(Image={'Bytes': image}, Attributes=["ALL"])