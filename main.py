import cv2
import numpy as np
import mediapipe as mp
from time import time
import threading
import facial_recognition as fr


############################################
# -------- SPEED CONFIG --------------------
############################################
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
PROCESS_EVERY_N_FRAMES = 1          # process every frame
FALL_PERSIST_SECONDS = 5
MATCH_THRESHOLD = 60                 # person tracking distance


############################################
# -------- STATE ---------------------------
############################################
frame_count = 0
people = {}
fall_history = {}
next_id = 1
MAX_HISTORY = 8


############################################
# -------- THREAD CAPTURE ------------------
############################################
class FastCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, INPUT_WIDTH)
        self.cap.set(4, INPUT_HEIGHT)

        self.ret = False
        self.frame = None

        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while True:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.cap.release()


############################################
# -------- PERSON ID LOGIC -----------------
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
        people[pid] = {"centroid": centroid, "state": "NORMAL", "fall_time": 0, "last_seen": time()}
        fall_history[pid] = []
        return pid

    best_match = None
    best_dist = 99999

    for pid, data in people.items():
        px, py = data["centroid"]
        dist = np.linalg.norm(np.array(centroid) - np.array((px, py)))
        if dist < best_dist:
            best_dist = dist
            best_match = pid

    if best_dist < MATCH_THRESHOLD:
        people[best_match]["centroid"] = centroid
        people[best_match]["last_seen"] = time()
        return best_match

    pid = next_id
    next_id += 1
    people[pid] = {"centroid": centroid, "state": "NORMAL", "fall_time": 0, "last_seen": time()}
    fall_history[pid] = []
    return pid


############################################
# --------- FALL LOGIC (YOUR LOGIC) --------
############################################
def detectFall(pid, lm):
    if not lm or len(lm) < 33:
        return False

    head_y = lm[0][1]
    shoulder_y = (lm[11][1] + lm[12][1]) / 2
    hip_y = (lm[23][1] + lm[24][1]) / 2
    knee_y = (lm[25][1] + lm[26][1]) / 2

    torso_length = abs(hip_y - shoulder_y)

    fall_history[pid].append({
        "head": head_y,
        "shoulder": shoulder_y,
        "hip": hip_y,
        "knee": knee_y,
        "torso": torso_length
    })

    if len(fall_history[pid]) > MAX_HISTORY:
        fall_history[pid].pop(0)

    if len(fall_history[pid]) < MAX_HISTORY:
        return False

    prev = fall_history[pid][0]
    curr = fall_history[pid][-1]

    shoulder_drop = curr["shoulder"] - prev["shoulder"]
    torso_collapse = (prev["torso"] - curr["torso"]) > 40
    horizontal_body = abs(curr["shoulder"] - curr["hip"]) < 25
    knee_bend = (curr["knee"] - prev["knee"]) > 50
    head_crash = (curr["head"] - prev["head"]) > 60 and curr["head"] >= prev["hip"]

    fall = (
        (shoulder_drop > 60 and torso_collapse and horizontal_body)
        or (knee_bend and shoulder_drop > 45)
        or head_crash
    )

    return fall


############################################
# -------- LANDMARK DRAW -------------------
############################################
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def draw_landmarks(frame, pose_landmarks, fallen=False):
    color = (0, 0, 255) if fallen else (0, 255, 0)
    h, w, _ = frame.shape

    for c in mp_pose.POSE_CONNECTIONS:
        s = pose_landmarks.landmark[c[0]]
        e = pose_landmarks.landmark[c[1]]
        cv2.line(frame, (int(s.x*w), int(s.y*h)),
                 (int(e.x*w), int(e.y*h)), color, 2)

    for lm in pose_landmarks.landmark:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, color, -1)


############################################
# -------- VIDEO LOOP ----------------------
############################################
frr = fr.FaceRecognition()
frr.encode_faces()

cam = FastCamera(0)

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark
        landmarks = [(int(l.x*w), int(l.y*h), l.z) for l in lm]

        centroid = get_centroid(landmarks)
        pid = assign_person_id(centroid)

        fell = detectFall(pid, landmarks)

        if fell and people[pid]["state"] != "FALLEN":
            people[pid]["state"] = "FALLEN"
            people[pid]["fall_time"] = time()

        if people[pid]["state"] == "FALLEN":
            if time() - people[pid]["fall_time"] < FALL_PERSIST_SECONDS:
                draw_landmarks(frame, results.pose_landmarks, True)
                cv2.putText(frame, f"ID {pid} - FALL DETECTED",
                            (20, 40+pid*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,0,255), 2)
            else:
                people[pid]["state"] = "NORMAL"
                draw_landmarks(frame, results.pose_landmarks, False)
        else:
            draw_landmarks(frame, results.pose_landmarks, False)

    # Cleanup disappeared people
    remove_list = [pid for pid,p in people.items() if time() - p["last_seen"] > 4]
    for pid in remove_list:
        del people[pid]
        fall_history.pop(pid, None)

    cv2.imshow("FAST CROWD FALL + FACE SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
