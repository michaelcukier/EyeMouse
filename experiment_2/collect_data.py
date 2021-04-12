import face_recognition
import cv2
import pyautogui
from datetime import datetime


count = 1000
while True:
    if count % 50 == 0:
        print(count)
    # now = datetime.now()
    #
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    count += 1
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    I = frame[:, :, ::-1]

    # I = face_recognition.load_image_file('./beautif.jpg')

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

    import numpy as np
    import cv2
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

    # Now we can crop again just the envloping rectangle
    finalImage = cropedImage[minY:maxY,minX:maxX]

    cursor_x, cursor_y = pyautogui.position()
    x = open("dlib_data/cursor-{}.txt".format(str(count) + '-' + str(cursor_x) + '-' + str(cursor_y)), "x")

    s1 = 'dlib_data/{}.jpg'.format(str(count) + '-0')
    cv2.imwrite(s1, finalImage)





