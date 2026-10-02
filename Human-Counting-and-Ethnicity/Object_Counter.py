import cv2
import warnings
from ultralytics import YOLO, solutions

warnings.filterwarnings('ignore')

# Set up the video capture with a local video file
cap = cv2.VideoCapture('video_7.mp4')  # replace with your video file path
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

    # Perform object tracking
    results = model.track(frame, persist=True, show=False)

    # Count objects
    frame = counter.start_counting(frame, results)

    # Write the processed frame to the output video
    video_writer.write(frame)

    # Display the frame with object counting
    cv2.imshow('Combined Video', frame)

    # Exit the loop if the user presses the 'Escape' key
    if cv2.waitKey(1) & 0xFF == 27:  # ASCII value of 'Escape' key is 27
        break

# Release the video capture and writer, and destroy all windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()
