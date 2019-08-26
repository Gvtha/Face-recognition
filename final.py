import cv2
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX

alist = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
glist = ['male', 'female']
MODEL_MEAN_VALUES = (200, 250, 300)
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
while 1:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        face_img = frame[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = alist[age_preds[0].argmax()]
        print("Age Range: " + age)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = glist[gender_preds[0].argmax()]
        print("Gender: " + gender)

        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
cv2.destroyAllWindows()
