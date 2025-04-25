# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:31:23 2024

@author: isis.bonet
"""

from ultralytics import YOLO
import cv2

model = YOLO('yolo11m.pt')

video_capture = cv2.VideoCapture(0)
while(True):
    ret, frame = video_capture.read()
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Nombre del objeto detectado
            # Dibujar el rectángulo y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 0, 0), 2)
    # Mostrar la imagen de resultado en el video
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == 27: #detener con tecla Esc
        break
video_capture.release()
cv2.destroyAllWindows()
