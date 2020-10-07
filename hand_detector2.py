import cv2
import numpy as np
import time
import math

kernel_size = 5

def convert(img):
    hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = (0,20,77)
    upper = (179,255,255)
    mask = cv2.inRange(hsv_image,lower,upper)
    blur = cv2.GaussianBlur(mask,(kernel_size,kernel_size),3)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size,kernel_size))
    closing = cv2.morphologyEx(blur,cv2.MORPH_CLOSE,kernel,iterations=1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening
    
def find_contours(binary,og_img):
    contours,heirarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        time.sleep(2)
        
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            cv2.drawContours(og_img,c,-1,(255,0,0),thickness=3)
    return contours

def boundrary(img,og_img):
    
    hull_list = []
    for i in range(len(img)):
        hull = cv2.convexHull(img[i],returnPoints=True)
        hull_list.append(hull)
    #for i in range(len(img)):                                          # draws convex hull
    #    cv2.drawContours(og_img,hull_list,i,(0,0,255),thickness=4)
    return hull_list    
    '''
    max_area = -1
    for i in range(len(cnt)):
        area = cv2.contourArea(cnt[i])
        if area > max_area:
            c = cnt[i]
            max_area = area
    c = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    hull = cv2.convexHull(c,returnPoints=False)
    defects = cv2.convexityDefects(c,hull)
    cv2.drawContours(og_img,c,-1,(0,0,255),thickness=4)
    cv2.drawContours(og_img,hull,-1,(255,0,255),thickness=4)
    return defects
    '''

def defects(cnts,hulls,og_img):
    max_area = -1
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > max_area:
            cnt_max = cnts[i]
            max_area = area
    hull_max = cv2.convexHull(cnt_max,returnPoints=False)
    defect = cv2.convexityDefects(cnt_max,hull_max)         #numpy array
    
    for i in range(defect.shape[0]):
        s,e,f,d = defect[i,0]
        start = tuple(cnt_max[s][0])
        end = tuple(cnt_max[e][0])
        far = tuple(cnt_max[f][0])
        print(start)
        cv2.line(og_img,start,end,[0,255,0],2)
        #cv2.circle(og_img,far,5,[0,0,255],-1)
    return defect,cnt_max

def cosine(A,B,C):
    angle_radian = math.acos((B**2 + C**2 - A**2)/(2*B*C))
    angle_degree = angle_radian * (180/3.14159)
    return angle_degree

def count_fingers(defect,cnt,og_img,frame):
    count = 0
    for i in range(defect.shape[0]):
        s,e,f,d = defect[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        EF = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        SF = math.sqrt((start[0] - far[0])**2 + (start[1] - far[1])**2) 
        ES = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

        if cosine(ES,EF,SF) < 90.0:
            cv2.circle(og_img,far,5,[0,0,255],-1)
            count += 1
        else:
            continue
    if count != 0:
        number = str(count + 1)
    elif count == 0:
        number = '0'
    display = cv2.putText(frame,number,(20,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)

def main():
    video_input = cv2.VideoCapture(0)
    while True:
        ret, frame = video_input.read()
        text = cv2.putText(frame,'Put hand in black box',(200,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
        roi = frame[100:300,100:300]
        cv2.rectangle(frame,(100,100),(300,300),(0,0,0),thickness=4)
        blur_frame = convert(roi)
        cnt_frame = find_contours(blur_frame,roi)
        hull_frame = boundrary(cnt_frame,roi)
        defect_frame, max_cnt_frame = defects(cnt_frame,hull_frame,roi)
        count_fingers(defect_frame,max_cnt_frame,roi,frame)
        
        cv2.imshow('frame1',frame)
        cv2.imshow('frame2',blur_frame)
        
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
    video_input.release()
    cv2.destroyAllWindows()