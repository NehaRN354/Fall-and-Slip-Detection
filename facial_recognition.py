import face_recognition
import os
import cv2
import numpy as np

class FaceRecognition:
    def __init__(self, image_folder="pictures"):
        self.known_faces = []
        self.known_face_names = []
        self.image_folder = image_folder

    def encode_faces(self):
        if not os.path.exists(self.image_folder):
            print("[ERROR] pictures folder not found.")
            return

        for image_name in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, image_name)

            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"[WARNING] No face found in {image_name}")
                continue

            self.known_faces.append(encodings[0])
            self.known_face_names.append(os.path.splitext(image_name)[0])

            print(f"[LOADED] {image_name}")

        print(f"[INFO] Total faces loaded: {len(self.known_faces)}")

    def recognize_face(self, frame):
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for encoding in encodings:
            distances = face_recognition.face_distance(self.known_faces, encoding)
            if len(distances) == 0:
                return None

            best_match = np.argmin(distances)

            if distances[best_match] < 0.5:
                return self.known_face_names[best_match]

        return None
