import cv2
import matplotlib.pyplot as plt
import streamlit as st
from deepface import DeepFace
from ultralytics import YOLO, solutions
import warnings
import tempfile
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Video Processing App", page_icon=":movie_camera:")

# Custom styling
st.markdown("""
<style>
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    flex-direction: column;
}

.header {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 20px;
    color: #333;
}

.sub-header {
    font-size: 1.5rem;
    margin-bottom: 30px;
    color: #555;
}

.btn-container {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

.btn {
    padding: 15px 30px;
    font-size: 1rem;
    font-weight: bold;
    text-align: center;
    border-radius: 5px;
    color: #fff;
    cursor: pointer;
    margin: 0 10px;
}

.btn-process {
    background-color: #007bff;
    border: none;
}

.btn-process:hover {
    background-color: #0056b3;
}

.btn-exit {
    background-color: #dc3545;
    border: none;
}

.btn-exit:hover {
    background-color: #bd2130;
}

.upload-container {
    margin-bottom: 20px;
    text-align: center;
}

.upload-btn {
    padding: 10px 20px;
    font-size: 1rem;
    font-weight: bold;
    text-align: center;
    border-radius: 5px;
    background-color: #6c757d;
    border: none;
    color: #fff;
    cursor: pointer;
}

.upload-btn:hover {
    background-color: #5a6268;
}

.video-container {
    margin-top: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Function to update region points
def update_region_points(position, w, h):
    return [
        (int(w * 0.02), position),
        (int(w * 0.98), position),
        (int(w * 0.98), position - int(h * 0.08)),
        (int(w * 0.02), position - int(h * 0.08))
    ]

# Function to process video for combined model
def process_combined_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    initial_position = int(h * 0.76)  # initial position as a percentage of video height
    region_points = update_region_points(initial_position, w, h)

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=region_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model.track(frame, persist=True, show=False)
        face_labels = {}

        for result in results:
            for track in result.boxes:
                x1, y1, x2, y2 = map(int, track.xyxy[0].numpy())
                track_id = track.id

                person_image = frame[y1:y2, x1:x2]
                person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(person_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (fx, fy, fw, fh) in faces:
                    fx += x1
                    fy += y1
                    face_image = frame[fy:fy+fh, fx:fx+fw]
                    result = DeepFace.analyze(face_image, actions=['race'], enforce_detection=False)
                    ethnicity_probs = result[0]['race']
                    dominant_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)

                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    face_labels[track_id] = dominant_ethnicity

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if track_id in face_labels:
                    cv2.putText(frame, face_labels[track_id], (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = counter.start_counting(frame, results)
        video_writer.write(frame)

        # Show the frame in a separate window
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'Escape' key is pressed
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Function to process video for ethnicity detection only
def process_ethnicity_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)), 
                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_image, actions=['race'], enforce_detection=False)
            ethnicity_probs = result[0]['race']
            dominant_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_ethnicity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        video_writer.write(frame)
        cv2.imshow('Ethnicity Detection Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'Escape' key is pressed
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Function to process video for human counting only
def process_human_counting_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    initial_position = int(h * 0.76)  # initial position as a percentage of video height
    region_points = update_region_points(initial_position, w, h)

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=region_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, show=False)
        frame = counter.start_counting(frame, results)
        video_writer.write(frame)
        cv2.imshow('Human Counting Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'Escape' key is pressed
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# Streamlit user interface
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="header">Video Processing App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a video to detect ethnicity and/or count humans</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="file_uploader")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.video(temp_video_path)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="btn-container">', unsafe_allow_html=True)
    if st.button("Process Ethnicity Detection Only", key="process_ethnicity"):
        output_path = os.path.join(tempfile.gettempdir(), "ethnicity_detection_video.mp4")
        process_ethnicity_video(temp_video_path, output_path)
        st.success("Ethnicity detection complete!")
        st.write(f"Processed video saved to: {output_path}")

    if st.button("Process Human Counting Only", key="process_human_counting"):
        output_path = os.path.join(tempfile.gettempdir(), "human_counting_video.mp4")
        process_human_counting_video(temp_video_path, output_path)
        st.success("Human counting complete!")
        st.write(f"Processed video saved to: {output_path}")

    if st.button("Process Combined Models", key="process_combined"):
        output_path = os.path.join(tempfile.gettempdir(), "combined_video.mp4")
        process_combined_video(temp_video_path, output_path)
        st.success("Combined model processing complete!")
        st.write(f"Processed video saved to: {output_path}")

    if st.button("Exit", key="exit"):
        st.stop()

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
