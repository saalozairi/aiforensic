import cv2
import numpy as np
import os

def load_and_encode_images(image_folder):
    face_encodings = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            encodings = encode_faces(image)
            if encodings:
                face_name = os.path.splitext(filename)[0]
                face_encodings[face_name] = encodings[0]
            else:
                print(f"No faces found in the image {filename}.")
    return face_encodings

# Function to encode faces in an image
def encode_faces(image):
    
    encoding = [np.random.rand(128)]  # Random 128-dimensional vector
    return encoding

# Function to save a new face image
def save_new_face(image, name, directory='known_faces'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"{directory}/{name}.jpg", image)

# Initialize video capture from webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Could not open video device")

# Initialize dictionaries to store known faces and new faces
known_faces = load_and_encode_images('known_faces')
new_faces = {}
face_trackers = {}
frame_number = 0

try:
    while True:
        ret, frame = video_capture.read()
        frame_number += 1
        if not ret:
            print("Failed to grab frame")
            break

        # Only process every other frame to save computational resources
        if frame_number % 2 == 0:
            face_location = (0, 0, frame.shape[0], frame.shape[1])
            face_encoding = encode_faces(frame)

            
            if known_faces:
                name = list(known_faces.keys())[0]  # Randomly assign a name from known faces
            else:
                name = f"new_face_{len(new_faces) + 1}"  # Generate a new face ID for unknown faces

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()

    # After closing the camera, ask user to name any new faces
    for face_id, face_info in face_trackers.items():
        if face_info['count'] > 4:  # Confirm face only if seen more than 4 times
            cv2.imshow("Save Face?", face_info['face_image'])
            cv2.waitKey(1)
            user_input = input(f"Enter name for {face_id} or press enter to skip: ").strip()
            if user_input:
                save_new_face(face_info['face_image'], user_input)
            cv2.destroyAllWindows()
