import numpy as np
import cv2
import pyautogui

face_cascade = cv2.CascadeClassifier('cv2-xml-cascade/face.xml')
eye_cascade = cv2.CascadeClassifier('cv2-xml-cascade/eye.xml')

cap = cv2.VideoCapture(0)
count = 0

DATASET_FOLDER = 'data-3700'

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
                s1 = '{0}/{1}-{2}.jpg'.format(DATASET_FOLDER, str(count), str(i))
                cv2.imwrite(s1, crop_img)
                i += 1
            f = open("{0}/head-{1}-{2}-{3}.txt".format(DATASET_FOLDER, str(count), str(x_loc), str(y_loc)), "x")
            cursor_x, cursor_y = pyautogui.position()
            x = open("{0}/cursor-{1}-{2}-{3}.txt".format(DATASET_FOLDER, str(count), str(cursor_x), str(cursor_y)), "x")
    count = count + 1
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()