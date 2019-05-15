
import cv2
from imutils import paths
import numpy as np
import imutils
from imageai.Detection import ObjectDetection
import os
import serial as sl
import time
print(cv2.__version__)
print('Starting')


ser = sl.Serial('/dev/cu.usbmodem144101',9600) 
time.sleep(1)

video_src = 'videos/rearR.mp4'

cap = cv2.VideoCapture(video_src)

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "data/resnet50_coco_best_v2.0.1.h5"))
detector.CustomObjects(car=True)
detector.loadModel( detection_speed='fastest')

anlayse_every_nth_frame = 10
n=0

capHeight = cap.get(4)
capWidth = cap.get(3)  # float

buildRegion = lambda x,to: {'from':x, 'to': to}
oneRegionWidth = capWidth/3

rightRegion = buildRegion(0,oneRegionWidth)
middleRegion = buildRegion(rightRegion['to'],oneRegionWidth*2)
leftRegion = buildRegion(middleRegion['to'],oneRegionWidth*3)

cap_area = capWidth*capHeight
seen =[]

signals = {'left': 0, 'middle': 1, 'right': 2,'leftM': 3, 'middleM': 4, 'rightM': 5,'leftS': 6, 'middleS': 7, 'rightS': 8, 'close': 9}

point_threshold_distance = 50

def is_in_region(region, x):
    return x > region['from'] and x < region['to']


def calc_distance(x1, x2):
    return abs(x1 - x2)

def is_in_point_threshold(x1,x2):
    if(calc_distance(x1,x2) > point_threshold_distance):
        return False
    return True

def is_taking_percantage_of_rectangle(threshold, width, height, cords):
    target_area=width*height
    taking_percentage_of_cap_rectangle = (target_area/cap_area)*100
    if(taking_percentage_of_cap_rectangle >threshold ):
        if(is_moving_forward(*cords)):
            return True
            


def middle(x1, x2):
    return x1 + calc_distance(x1,x2)/2


def classifie_to_region(middleXCoordinate):
    if(is_in_region(leftRegion, middleXCoordinate)):
        print('LEFT')
        ser.write(str(signals['left']).encode())
        return 'LEFT'
    if(is_in_region(middleRegion, middleXCoordinate)):
        ser.write(str(signals['middle']).encode())
        print('MIDDLE')
        return 'MIDDLE'
    if(is_in_region(rightRegion, middleXCoordinate)):
        ser.write(str(signals['right']).encode())
        print('RIGHT')
        return 'RIGHT'



def area(w,h):
    return w*h

def is_moving_forward(x,y, x1,y1):
    for (i,cords) in enumerate(seen):
        [cx,cy, cx1, cy1] = cords
        if(is_in_point_threshold(x,cx) and is_in_point_threshold(y,cy)):
            target_width = calc_distance(x,x1)
            target_height = calc_distance(y,y1)
            prev_width = calc_distance(cx,cx1)
            prev_height = calc_distance(cy,cy1)
            if(area(prev_width,prev_height) < area(target_height,target_width)):
                return True



while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    if(n < anlayse_every_nth_frame):
        n+=1
        cv2.imshow('video', img)
        continue

    n = 0
    detections = detector.detectObjectsFromImage(input_image=img, input_type='array',  output_type='array')
    vehicles = detections[1]

    for (v) in vehicles:
        detection_name = v['name']
        [x1,y1,x2,y2] = v['box_points']
        print(x1, x2)
        cords = [x1,y1,x2,y2]
        seen.append(cords)
        if(is_taking_percantage_of_rectangle(3,abs(x1-x2), abs(y1-y2), [x1,y1,x2,y2])):
            cv2.putText(img, 'Warning', (x1 , y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),thickness=2) 
        detectionRegion = classifie_to_region(middle(x1,x2))
        cv2.putText(img, detectionRegion, (int(middle(x1,x2)) , int(middle(y1,y2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 248, 0),thickness=2) 
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)  

    new_seen = []
    for (i,cords) in enumerate(seen):
        [cx,cy, cx1, cy1] = cords
        for (v) in vehicles:
            [x1,y1, x2,y2] = v['box_points']
            if(is_in_point_threshold(x1,cx) and is_in_point_threshold(y1,cy)):
                new_seen.append(cords)

    seen = new_seen   
    cv2.imshow('video', img)
 
    if cv2.waitKey(33) == 27:
        ser.write(str(signals['close']).encode())
        break

cv2.destroyAllWindows()


