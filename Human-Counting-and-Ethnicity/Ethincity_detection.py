import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Load the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the video capture with a local video file
cap = cv2.VideoCapture('video_6.mp4')  # replace with your video file path

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Crop the face region
        face_image = frame[y:y+h, x:x+w]

        # Analyze the ethnicity of the face
        result = DeepFace.analyze(face_image, actions=['race'],enforce_detection=False)

        # Extract the ethnicity probabilities
        ethnicity_probs = result[0]['race']

        # Find the dominant ethnicity
        dominant_ethnicity = max(ethnicity_probs, key=ethnicity_probs.get)

        # Draw a rectangle around the face and label it with the dominant ethnicity
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_ethnicity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit the loop if the user presses the 'Escape' key
    if cv2.waitKey(1) & 0xFF == 27:  # ASCII value of 'Escape' key is 27
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
