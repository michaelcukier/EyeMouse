import face_recognition
import pyautogui
import numpy as np
import cv2

DATASET_FOLDER = 'dlib_data'
count = 1000

while True:
    if count % 50 == 0:
        print(count)
    count += 1
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    I = frame[:, :, ::-1]
    face_landmarks = face_recognition.face_landmarks(I)
    landmark = face_landmarks[0].get('left_eye')
    minX = I.shape[1]
    maxX = -1
    minY = I.shape[0]
    maxY = -1
    for point in landmark:
        x = point[0]
        y = point[1]
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y
    # Go over the points in the image if thay are out side of the emclosing rectangle put zero
    # if not check if thay are inside the polygon or not
    cropedImage = np.zeros_like(I)
    for y in range(0,I.shape[0]):
        for x in range(0, I.shape[1]):
            if x < minX or x > maxX or y < minY or y > maxY:
                continue
            if cv2.pointPolygonTest(np.asarray(landmark),(x,y),False) >= 0:
                cropedImage[y, x, 0] = I[y, x, 0]
                cropedImage[y, x, 1] = I[y, x, 1]
                cropedImage[y, x, 2] = I[y, x, 2]
    finalImage = cropedImage[minY:maxY,minX:maxX]
    cursor_x, cursor_y = pyautogui.position()
    x = open("{0}/cursor-{1}-{2}-{3}.txt".format(DATASET_FOLDER, str(count), str(cursor_x), str(cursor_y)), "x")
    s1 = '{0}/{1}.jpg'.format(DATASET_FOLDER, str(count) + '-0')
    cv2.imwrite(s1, finalImage)





