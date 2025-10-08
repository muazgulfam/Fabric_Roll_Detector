# 🧮 Roll Counter Dashboard (YOLO + SORT + Streamlit + Oracle)

This project is a **real-time roll counting system** that uses **YOLO object detection** and **SORT tracking** to count rolls crossing a virtual line in a live camera feed. It features:

- 🎯 Real-time YOLO object detection  
- 🧠 SORT tracking with unique ID assignment  
- 📈 Automatic line crossing detection (forward & reverse)  
- 🖥️ Streamlit dashboard with real-time visualization  
- 🗃️ Oracle Database integration to save counts and related metadata  
- 📹 RTSP camera stream support  

---

## 🧰 Tech Stack

| Component             | Technology Used                            |
|-----------------------|--------------------------------------------|
| Object Detection       | [YOLO (:contentReference[oaicite:0]{index=0})](https://docs.ultralytics.com/)         |
| Tracking               | SORT (Kalman Filter based)                |
| UI Framework           | [:contentReference[oaicite:1]{index=1}](https://streamlit.io/)                              |
| Database               | [:contentReference[oaicite:2]{index=2}](https://www.oracle.com/database/)             |
| Language               | Python 3.8+                               |
| CV Library             | [:contentReference[oaicite:3]{index=3}](https://opencv.org/)                                 |

---

## 📦 Folder Structure

