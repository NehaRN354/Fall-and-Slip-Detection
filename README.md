# Fall-and-Slip-Detection
A robust AI-powered fall detection system designed for:

🏥 Hospitals & Elderly Care

🏭 Industrial & Workplace Safety

🏫 Schools & Public Buildings

🏠 Smart Homes

🚶‍♂️ Public Monitoring Zones

🧑‍🤝‍🧑 Crowded Environments

The system continuously monitors people, detects sudden collapse or unsafe posture transitions, and instantly raises alerts.


🎯 Key Features

✔ Real-Time Fall Detection 

✔ Multiple People Tracking

✔ High Sensitivity Mode (Demo-Friendly)

✔ Advanced Pose Landmark Analysis

✔ No Bounding Boxes – Skeleton Based

✔ Red Body Landmark Highlight on Fall

✔ Persistent Fall Detection Memory

✔ Face Recognition Support (Optional)

✔ Works With Webcam, CCTV, and Phone Camera Streams


🧠 How It Works (Simple Explanation)

Instead of basic bounding box detection, our system analyzes human biomechanics:

Sudden shoulder drop

Rapid head downward movement

Torso collapse

Knee buckling

Body transitioning horizontal

If these patterns match a fall → Fall Detected 🔴
Otherwise → Normal Movement 🟢

This makes it more reliable for real environments.

🧰 Tech Stack
Component	Technology

Pose Estimation	: MediaPipe

Computer Vision :	OpenCV

Math/Processing : NumPy

Face ID	: face_recognition (optional)

Language	: Python


⚙️ Installation Guide

1️⃣ Clone Repo
git clone https://github.com/YOUR_USERNAME/Smart-Fall-Detection.git
cd Smart-Fall-Detection

2️⃣ Create Virtual Environment
python -m venv mediapipe_env

3️⃣ Activate

Windows:

mediapipe_env\Scripts\activate


Mac/Linux:

source mediapipe_env/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run The System

Laptop Webcam

python main.py


Optional — Phone / CCTV Stream
Change source in code:

PHONE_STREAM = "http://YOUR_IP:PORT/video"

🧪 Output

🟢 Green Skeleton → Normal
🔴 Red Skeleton + Text → Fall Detected

Supports:
✔ Single Person
✔ Multiple People
✔ Moving Crowd

📂 Face Recognition (Optional)

Place images inside:

known/
   person1.jpg
   person2.jpg


Filename = Identity Name

System will:

Detect face

Assign name to skeleton

Use for categorization or logging

🌍 Real-World Applications
Sector	Usage

Healthcare :	Elderly safety, ICU monitoring

Industrial :	Worker safety, accident prevention

Smart Buildings :	Fall alerts in offices, campuses

Home Automation :	Elderly living alone

Public Monitoring	: Metro stations, malls, airports

Rehabilitation : Stroke & injury recovery


🚀 Future Enhancements

🔹 SMS / Email Alerts

🔹 Cloud Integration (AWS / Azure)

🔹 Mobile App Dashboard

🔹 Fall Logging & Analytics

🔹 CCTV Optimized Fall Detection Mode

🔹 Reduced False Positives With ML Model
