import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while (True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    imagem = frame
    fonte = cv2.FONT_HERSHEY_SIMPLEX

    if len(faces) > 0:
        plural = ''
        cont_faces = len(faces)
        if cont_faces > 1:
            plural = 's'
        text_faces = str(cont_faces) + ' face' + plural + ' detectada' + plural
        cv2.putText(imagem, text_faces, (10, 45), fonte, 1, (255, 0, 0), None, None)
        cv2.putText(imagem, str(len(eyes)), (10, 75), fonte, 1, (255, 0, 0), None, None)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
