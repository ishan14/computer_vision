import numpy as np
import cv2

video_input = cv2.VideoCapture(0)

myColors = [[40,80,131,101,255,255],
            [128,88,178,179,228,255]]

myColorValues = [[0,128,0],
                 [203,192,255]]

myPoints = []      ##[x,y,colorID]

def findcolor(image):

    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(hsv_image,lower,upper)
        x,y = getContours(mask)
        cv2.circle(imageResult,(x,y),10,myColorValues[count],cv2.FILLED)
        if x!= 0 and y!=0:
            newPoints.append([x,y,count])
        print(count)
        count += 1
        
        
    return newPoints


def getContours(img):
    contours, heirarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    background = np.zeros([img.shape[0],img.shape[1],3],'uint8')
    x,y,w,h = 0,0,0,0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            #cv2.drawContours(imageResult,c,-1,(255,0,0),3)
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
    return x+w//2,y

def DrawOnCanvas(myPoints):
    for point in myPoints:
        cv2.circle(imageResult,(point[0],point[1]),5,myColorValues[point[2]],cv2.FILLED)

        

while True:
    ret, image = video_input.read()
    imageResult = image.copy()
    newPoints = findcolor(image)
    if len(newPoints)!= 0:
        for newP in newPoints:
            myPoints.append(newP)
    
    if len(myPoints)!=0:
        DrawOnCanvas(myPoints)
    cv2.imshow("Frame",imageResult)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
video_input.release()
cv2.destroyAllWindows()