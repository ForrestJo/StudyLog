
import numpy as np
import csv
import os


#把所有的图片读入内存
def read_images(csvfile):
    path = os.getcwd()
    csvfile = open(csvfile, 'r')
    reader = csv.reader(csvfile)
    X, y = [], []
    for row in reader:
        img = cv2.imread(os.path.join(path + row[0]), cv2.IMREAD_GRAYSCALE)
        X.append(np.asarray(img, dtype = np.uint8))
        y.append(int(row[1])); 
    csvfile.close; return [X, y]

[X, y] = read_images('../

#训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(np.asarray(X), np.asarray(y))

while cv2.waitKey(10) == -1:
    face_cascade = cv2.CascadeClassifier('/media/nvidia/gmdisk/1/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, h + y), (255, 0, 0), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200),interpolation = cv2.INTER_LINEAR)
        params = model.predict(roi)
        print "Label: %s, Confidence: %.2f" %(params[0], params[1])
        cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow('camera', img); success, frame = camera.read()
else:
    cv2.destroyAllWindows()














