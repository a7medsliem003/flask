import cv2
import numpy as np
import face_recognition
import os
from flask import Flask, request, jsonify
from threading import Thread

app = Flask(__name__)

# تحميل الوجوه المعروفة
path = 'persons'
images = []
classNames = []

personsList = os.listdir(path)

for filename in personsList:
    img = cv2.imread(f'{path}/{filename}')
    images.append(img)
    classNames.append(os.path.splitext(filename)[0])

# دالة لتحويل الصور إلى Encodings
def findEncoding(images):
    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

# استخراج الـ encodings والأسماء
known_face_encodings = findEncoding(images)
known_face_names = classNames

print("Encoding complete.")

url='http://192.168.4.1/cam-hi.jpg'

# دالة API للتعرف على الوجوه من صورة
@app.route('/identify', methods=['POST'])
def identify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = face_recognition.load_image_file(file)

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        names.append(name)

    return jsonify({'people': names})

# دالة لتشغيل الكاميرا وعرض الوجوه المتعرف عليها في الوقت الحقيقي
def run_camera():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(imgS)
        face_encodings = face_recognition.face_encodings(imgS, face_locations)

        for encodeface, faceloc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encodeface)
            faceDis = face_recognition.face_distance(known_face_encodings, encodeface)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = known_face_names[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# تشغيل الكاميرا في thread منفصل
def start_camera_thread():
    Thread(target=run_camera).start()

if __name__ == '__main__':
    # تشغيل الكاميرا في بداية التطبيق
    start_camera_thread()
    
    # تشغيل السيرفر
    app.run(host='0.0.0.0', port=5000)
