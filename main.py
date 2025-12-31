import cv2
import numpy as np
import mediapipe as mp
from time import time
import threading
import face_recognition
import os

############################################
# CONFIG
############################################
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
FALL_PERSIST_SECONDS = 1.5
MATCH_THRESHOLD = 60
MAX_HISTORY = 8

people = {}
fall_history = {}
cctv_history = {}
next_id = 1

############################################
# LOAD KNOWN FACES
############################################
known_faces = []
known_names = []

def load_known_faces():
    folder = "known"
    if not os.path.exists(folder):
        print("known folder not found")
        return

    for image in os.listdir(folder):
        path = os.path.join(folder, image)

        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)

        if len(enc) == 0:
            continue

        known_faces.append(enc[0])
        known_names.append(image.split(".")[0].lower())

    print("Faces Loaded:", known_names)

def is_known(frame):
    small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for enc in encodings:
        matches = face_recognition.compare_faces(known_faces, enc)
        if True in matches:
            return True

    return False


############################################
# CAMERA THREAD
############################################
class FastCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, INPUT_WIDTH)
        self.cap.set(4, INPUT_HEIGHT)

        self.ret = False
        self.frame = None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.cap.release()


############################################
# PERSON TRACKING
############################################
def get_centroid(lm):
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def assign_person_id(centroid):
    global next_id

    if not people:
        pid = next_id
        next_id += 1
        people[pid] = {"centroid": centroid, "state": "NORMAL", "fall_time": 0}
        fall_history[pid] = []
        cctv_history[pid] = []
        return pid

    best_match = None
    best_dist = 1e9

    for pid, data in people.items():
        px, py = data["centroid"]
        dist = np.linalg.norm(np.array(centroid) - np.array((px, py)))
        if dist < best_dist:
            best_dist = dist
            best_match = pid

    if best_dist < MATCH_THRESHOLD:
        people[best_match]["centroid"] = centroid
        return best_match

    pid = next_id
    next_id += 1
    people[pid] = {"centroid": centroid, "state": "NORMAL", "fall_time": 0}
    fall_history[pid] = []
    cctv_history[pid] = []
    return pid


############################################
# -------- FRONT FALL LOGIC ----------------
############################################
def detect_front_fall(pid, lm):
    head = lm[0][1]
    shoulder = (lm[11][1] + lm[12][1]) / 2
    hip = (lm[23][1] + lm[24][1]) / 2
    knee = (lm[25][1] + lm[26][1]) / 2

    torso = abs(hip - shoulder)

    fall_history[pid].append({
        "head": head,
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "torso": torso
    })

    if len(fall_history[pid]) > MAX_HISTORY:
        fall_history[pid].pop(0)

    if len(fall_history[pid]) < MAX_HISTORY:
        return False

    prev = fall_history[pid][0]
    curr = fall_history[pid][-1]

    shoulder_drop = curr["shoulder"] - prev["shoulder"]
    torso_collapse = (prev["torso"] - curr["torso"]) > 40
    horizontal = abs(curr["shoulder"] - curr["hip"]) < 25
    knee_bend = (curr["knee"] - prev["knee"]) > 50
    head_crash = (curr["head"] - prev["head"]) > 60 and curr["head"] >= prev["hip"]

    return (
        (shoulder_drop > 60 and torso_collapse and horizontal)
        or (knee_bend and shoulder_drop > 45)
        or head_crash
    )


############################################
# -------- CCTV FALL LOGIC -----------------
############################################
def detect_cctv_fall(pid, lm):
    if len(lm) < 33:
        return False

    head = lm[0][1]
    shoulder = (lm[11][1] + lm[12][1]) / 2
    hip = (lm[23][1] + lm[24][1]) / 2
    knee = (lm[25][1] + lm[26][1]) / 2

    torso = abs(hip - shoulder)
    if torso <= 0:
        return False

    cctv_history[pid].append({
        "torso": torso,
        "head": head,
        "shoulder": shoulder,
        "knee": knee
    })

    if len(cctv_history[pid]) > MAX_HISTORY:
        cctv_history[pid].pop(0)

    if len(cctv_history[pid]) < MAX_HISTORY:
        return False

    prev = cctv_history[pid][0]
    curr = cctv_history[pid][-1]

    torso_ratio = curr["torso"] / prev["torso"]

    torso_flat = torso_ratio < 0.55
    head_drop = (curr["head"] - prev["head"]) > 45
    shoulder_drop = (curr["shoulder"] - prev["shoulder"]) > 40
    knee_rise = (prev["knee"] - curr["knee"]) > 20

    if knee_rise:
        return False

    return torso_flat and head_drop and shoulder_drop


############################################
# MEDIAPIPE MULTI-PERSON POSE
############################################
model_path = "models/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=5
)

pose_detector = PoseLandmarker.create_from_options(options)


############################################
# DRAW
############################################
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def draw(frame, person_landmarks, fallen=False):
    color = (0, 0, 255) if fallen else (0, 255, 0)
    h, w, _ = frame.shape

    for lm in person_landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, color, -1)

    for c in POSE_CONNECTIONS:
        s = person_landmarks[c[0]]
        e = person_landmarks[c[1]]
        cv2.line(frame,
                 (int(s.x*w), int(s.y*h)),
                 (int(e.x*w), int(e.y*h)),
                 color, 2)


############################################
# MAIN
############################################
load_known_faces()
cam = FastCamera(0)
frame_index = 0

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    known = is_known(frame)

    label = "KNOWN" if known else "UNKNOWN"
    col = (0,255,0) if known else (0,0,255)

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    result = pose_detector.detect_for_video(mp_image, frame_index)
    frame_index += 1

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            lm = [(int(p.x*w), int(p.y*h), p.z) for p in pose_landmarks]

            centroid = get_centroid(lm)
            pid = assign_person_id(centroid)

            front = detect_front_fall(pid, lm)
            cctv = detect_cctv_fall(pid, lm)

            fell = front or cctv

            if fell:
                people[pid]["state"] = "FALLEN"
                people[pid]["fall_time"] = time()

            fallen = (
                people[pid]["state"] == "FALLEN" and
                time() - people[pid]["fall_time"] < FALL_PERSIST_SECONDS
            )

            draw(frame, pose_landmarks, fallen)

            if fallen:
                cv2.putText(frame, "FALL DETECTED",
                            (centroid[0], centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

    cv2.imshow("MULTI PERSON FALL DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
