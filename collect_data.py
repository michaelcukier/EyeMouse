import numpy as np
import cv2
import pyautogui
face_cascade = cv2.CascadeClassifier('face2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
count=1

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        x_loc = x
        y_loc = y

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            i = 0
            for (ex,ey,ew,eh) in eyes:

                crop_img = roi_color[ey: ey + eh, ex: ex + ew]
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                s1 = 'tmp/{}.jpg'.format(str(count) + '-' + str(i))
                cv2.imwrite(s1, crop_img)
                i += 1
            f = open("tmp/head-{}.txt".format(str(count) + '-' + str(x_loc) + '-' + str(y_loc)), "x")
            cursor_x, cursor_y = pyautogui.position()
            x = open("tmp/cursor-{}.txt".format(str(count) + '-' + str(cursor_x) + '-' + str(cursor_y)), "x")
        # quit()
    count = count + 1
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()