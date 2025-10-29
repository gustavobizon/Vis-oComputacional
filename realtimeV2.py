import cv2
import keras
import numpy as np

model =keras.models.load_model ("melhor_modelo.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    boolRetorno, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3,5)
    try: 
        for (x1, y1, r, s) in faces:
            image = gray[y1:y1+s, x1:x1+r]
            cv2.rectangle(frame, (x1, y1),(x1+r, y1+s), (255, 0, 0), 2)
            image = cv2.resize(image,(48, 48))
            features = extract_features(image)
            pred = model.predict(features)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(frame, '% s' %(prediction_label), (x1-10, y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            cv2.imshow("Output", frame)
            cv2.waitKey(27)
    except cv2.error:
        pass