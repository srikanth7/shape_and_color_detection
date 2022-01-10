import cv2
import numpy as np

def detector(img):
    detected_shapes = []
    imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgGry, (1, 1), cv2.BORDER_DEFAULT)
    equ = cv2.equalizeHist(blur)
    ret , thrash = cv2.threshold(equ, 240 , 255, cv2.CHAIN_APPROX_NONE)
    contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (0, 0, 0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])   # calculates Moments
            cy = int(M['m01']/M['m00'])
        px0 = img[ cy, cx, 0]
        px1 = img[ cy, cx, 1]
        px2 = img[ cy, cx, 2]

        color = 'White'

        if(px0==255 and px1==0 and px2==0):
            color = 'Blue'
        elif(px0==0 and px1==0 and px2==255):
            color = 'Red'
        elif(px0==0 and px1==255 and px2==0):
            color = 'Green'
        elif(px0==0 and (px1==140 or px1==150) and px2==255):
            color = 'Orange'
        
        if len(approx) == 3:
            detected_shapes.append([color,'Triangle', (cx, cy)])
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, "Triangle", (cx-20, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            
            if 0.95 <= aspectRatio <= 1.05:
                detected_shapes.append([color,'Square',(cx, cy)])
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(img, "Square", (cx-20, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                detected_shapes.append([color,'Rectangle',(cx, cy)])
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(img, "Rectangle", (cx-20, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 5:
            detected_shapes.append([color,'Pentagon',(cx, cy)])
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, "Pentagon", (cx-20, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            detected_shapes.append([color,'Circle',(cx, cy)])
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, "Circle", (cx-20, cy-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    for detected in detected_shapes:
        color = detected[0]
        shape = detected[1]
        coordinates = detected[2]
        cv2.putText(img, str((color,shape)), coordinates, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)
    
    cv2.imshow('shapes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_shapes

def main():
    for i in range(1,16):
        img = cv2.imread("test_images/test_image_"+str(i)+".png")
        detected_shapes = detector(img)
        print("For Test_image_"+str(i)+".png: ")
        print(detected_shapes)
main()