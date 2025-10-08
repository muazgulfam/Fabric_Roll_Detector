import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from sort import Sort
import cx_Oracle
import threading
import time

# ======================================
# 🔹 ORACLE CONNECTION
# ======================================
try:
    cx_Oracle.init_oracle_client()
except cx_Oracle.ProgrammingError:
    pass  # already initialized

conn = cx_Oracle.connect(
    user="APPS",
    password="apps",
    dsn="10.0.145.41:1521/CLTEST"
)
print("Connected:", conn.version)
cursor = conn.cursor()

# ======================================
# 🔹 STREAMLIT CONFIGURATION
# ======================================
st.set_page_config(page_title="Roll Counter", layout="wide")

# Session states
if "total_counter" not in st.session_state:
    st.session_state.total_counter = 0
if "roll_states" not in st.session_state:
    st.session_state.roll_states = {}
if "prev_positions" not in st.session_state:
    st.session_state.prev_positions = {}

# ======================================
# 🔹 YOLO + SORT SETUP
# ======================================
model = YOLO(r"C:\Users\m.umair\Downloads\rollbest.pt")
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.1)

# Camera
IP_CAMERA_URL = "rtsp://admin:admin123@192.168.28.156:554/cam/realmonitor?channel=1&subtype=0"

# ======================================
# 🔹 GLOBAL VARIABLES for THREAD
# ======================================
latest_frame = None
stop_thread = False

# ======================================
# 🔹 CAPTURE THREAD FUNCTION
# ======================================
def capture_frames():
    global latest_frame, stop_thread
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        latest_frame = frame
    cap.release()

# Start capture thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

# ======================================
# 🔹 LINE CONFIG (Original full-res coordinates)
# ======================================
line_start = (2559, 693)
line_end   = (2308, 636)

def point_side_of_line(p, a, b):
    """Return >0 if left, <0 if right, 0 if on the line"""
    return (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])

# ======================================
# 🔹 STREAMLIT UI
# ======================================
st.title("🎥 Real-time Roll Counter Dashboard")

video_col, info_col = st.columns([2, 1])

with info_col:
    st.markdown("## 📊 Counter & Inputs")

    total_placeholder = st.empty()
    total_placeholder.metric("Total Rolls", st.session_state.total_counter)

    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 310px !important;
            font-weight: 800 !important;
            color: #00BFFF;
        }
        [data-testid="stMetricLabel"] {
            font-size: 28px !important;
            font-weight: 1000 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    license_plate = st.text_input("🚗 License Plate", key="license_plate")
    unit = st.text_input("🏭 Unit", key="unit")
    gate = st.text_input("🚧 Gate", key="gate")

    if st.button("🔄 Reset Counter"):
        try:
            cursor.execute("""
                INSERT INTO AGI_AI_ROLL_COUNT (LICENSE_PLATE, UNIT, GATE, ROLL_COUNT)
                VALUES (:1, :2, :3, :4)
            """, (st.session_state.license_plate,
                  st.session_state.unit,
                  st.session_state.gate,
                  st.session_state.total_counter))
            conn.commit()
            st.success("✅ Data saved to Oracle successfully before reset!")

            st.session_state.total_counter = 0
            st.session_state.roll_states = {}
            st.session_state.prev_positions = {}
            st.session_state.license_plate = ""
            st.session_state.unit = ""
            st.session_state.gate = ""
            st.success("Counter and inputs reset!")

        except Exception as e:
            st.error(f"❌ Error while saving: {e}")

frame_window = video_col.empty()

# ======================================
# 🔹 LIVE PROCESSING LOOP
# ======================================
frame_id = 0
fps_limit = 0.03  # ~30 FPS display
last_detections = np.empty((0, 5))

while True:
    if latest_frame is None:
        st.warning("⏳ Waiting for camera feed...")
        time.sleep(0.5)
        continue

    frame = latest_frame.copy()
    frame_id += 1

    # ✅ Resize for YOLO
    resized_frame = cv2.resize(frame, (1280, 720))

    # ✅ Alternate frame prediction (for performance)
    if frame_id % 2 == 0:
        results = model.predict(resized_frame, imgsz=640, conf=0.35, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, conf])
        detections = np.array(detections, dtype=float)
        last_detections = detections
    else:
        detections = last_detections

    if detections.ndim == 1:
        if detections.size == 0:
            detections = np.empty((0, 5))
        else:
            detections = detections.reshape(1, 5)

    tracks = tracker.update(detections) if len(detections) else []

    # ======================================
    # 🔹 LINE CROSSING LOGIC
    # ======================================
    for x1, y1, x2, y2, tid in tracks:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        if tid in st.session_state.prev_positions:
            prev = st.session_state.prev_positions[tid]
            prev_side = point_side_of_line(prev, line_start, line_end)
            curr_side = point_side_of_line((cx, cy), line_start, line_end)

            if prev_side * curr_side < 0:  # ✅ Line crossed
                if curr_side < prev_side:
                    if st.session_state.roll_states.get(tid) != "forward":
                        st.session_state.total_counter += 1
                        st.session_state.roll_states[tid] = "forward"
                        print(f"➡️ Roll {int(tid)} Forward | Total:{st.session_state.total_counter}")
                else:
                    if st.session_state.roll_states.get(tid) != "reverse":
                        st.session_state.total_counter -= 1
                        st.session_state.roll_states[tid] = "reverse"
                        print(f"⬅️ Roll {int(tid)} Reverse | Total:{st.session_state.total_counter}")

        st.session_state.prev_positions[tid] = (cx, cy)

        # 🟢 Draw box + ID + center
        cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(resized_frame, (cx, cy), 5, (0,0,255), -1)
        cv2.putText(resized_frame, f"ID {int(tid)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ======================================
    # 🟥 FIXED: Properly Scaled Virtual Line
    # ======================================
    orig_w, orig_h = frame.shape[1], frame.shape[0]
    draw_w, draw_h = 1280, 720
    scale_x = draw_w / orig_w
    scale_y = draw_h / orig_h

    scaled_start = (int(line_start[0] * scale_x), int(line_start[1] * scale_y))
    scaled_end   = (int(line_end[0] * scale_x), int(line_end[1] * scale_y))

    cv2.line(resized_frame, scaled_start, scaled_end, (0, 0, 255), 4)
    cv2.putText(resized_frame, f"Total Rolls: {st.session_state.total_counter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    # ======================================
    # 🔹 UPDATE STREAMLIT UI
    # ======================================
    total_placeholder.metric("Total Rolls", st.session_state.total_counter)
    frame_window.image(resized_frame, channels="BGR", use_container_width=True)

    time.sleep(fps_limit)
            