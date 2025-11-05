import face_recognition
import cv2
import os
import numpy as np

# Folder with known faces
KNOWN_FACES_DIR = "known_faces"

# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png")):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Pick best match
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
