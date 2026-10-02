import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from ultralytics import YOLO, solutions
import warnings

warnings.filterwarnings('ignore')

# Load the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the video capture with a local video file
cap = cv2.VideoCapture('video.mp4')  # replace with your video file path
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Define initial vertical position for object counting line
initial_position = int(h * 0.76)  # initial position as a percentage of video height

# Function to update region points
def update_region_points(position):
    global region_points
    region_points = [
        (int(w * 0.02), position),
        (int(w * 0.98), position),
        (int(w * 0.98), position - int(h * 0.08)),
        (int(w * 0.02), position - int(h * 0.08))
    ]

# Initialize region points with the initial position
update_region_points(initial_position)

# Video writer
video_writer = cv2.VideoWriter("combined_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform object tracking
    results = model.track(frame, persist=True, show=False)

    # Prepare a dictionary to hold face labels for each person ID
    face_labels = {}

    for result in results:
        for track in result.boxes:
            # Extract bounding box coordinates for the person
            x1, y1, x2, y2 = map(int, track.xyxy[0].numpy())
            track_id = track.id

            # Crop the person region
            person_image = frame[y1:y2, x1:x2]

            # Convert the person region to grayscale for face detection
            person_gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the person region
            faces = face_cascade.detectMultiScale(person_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Loop through the detected faces for ethnicity analysis
            for (fx, fy, fw, fh) in faces:
                # Adjust face coordinates to the original frame
                fx += x1
                fy += y1

                # Crop the face region
                face_image = frame[fy:fy+fh, fx:fx+fw]

                # Analyze the ethnicity of the face
                result = DeepFace.analyze(face_image, actions=['race'], enforce_detection=False)

                # Extract the ethnicity probabilities
                ethnicity_probs = result[0]['race']

                # Find the dominant ethnicity
                dominant_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

                # Store the dominant ethnicity for this person ID
                face_labels[track_id] = dominant_ethnicity

            # Draw a rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # If a face label exists for this person ID, add it above the person bounding box
            if track_id in face_labels:
                cv2.putText(frame, face_labels[track_id], (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Count objects
    frame = counter.start_counting(frame, results)

    # Write the processed frame to the output video
    video_writer.write(frame)

    # Display the frame with face and object tracking
    cv2.imshow('Combined Video', frame)

    # Exit the loop if the user presses the 'Escape' key
    if cv2.waitKey(1) & 0xFF == 27:  # ASCII value of 'Escape' key is 27
        break

# Release the video capture and writer, and destroy all windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()
