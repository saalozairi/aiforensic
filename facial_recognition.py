import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Convert image to grayscale
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load and encode known faces
def load_and_encode_images(image_folder):
    face_encodings = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            face_image = cv2.imread(image_path)  
            face_image_gray = to_grayscale(face_image)  
            encoding = encode_face(face_image_gray)  
            if encoding is not None:
                face_name = os.path.splitext(filename)[0]
                face_encodings[face_name] = encoding
            else:
                print(f"No face found in the image {filename}.")
    return face_encodings

# encode a single face
def encode_face(face_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(face_image, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 1:  
        (x, y, w, h) = faces[0]
        face_roi = face_image[y:y+h, x:x+w] 
        face_encoding = cv2.resize(face_roi, (128, 128)) 
        return face_encoding, (x, y, w, h)
    else:
        return None, (0, 0, 0, 0)  

# recognize faces in the video stream
def recognize_faces(frame, known_faces):
    frame_gray = to_grayscale(frame)  # convert the frame to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)
    print(f"Detected {len(faces)} faces in the frame")  
    for (x, y, w, h) in faces:
        face_roi = frame_gray[y:y+h, x:x+w]
        face_encoding = cv2.resize(face_roi, (128, 128))
        # Compare with known faces
        min_distance = float('inf')
        found_name = "Unknown"
        if face_encoding is not None:
            for name, known_encoding in known_faces.items():
                distance = np.linalg.norm(face_encoding - known_encoding[0])
                print(f"Distance to {name}: {distance}")
                if distance < min_distance:  
                    min_distance = distance
                    found_name = name
        else:
            print("No valid face encoding extracted.")
        # draw rectangle and name on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, found_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        if found_name == "Unknown" and is_new_face(face_roi):
            save_new_face(face_roi)


# save a new face image
def save_new_face(face_image, directory='known_faces'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"{directory}/new_face.jpg", face_image)
    new_face_name = input("New face detected. Please provide a name: ")
    if new_face_name:
        os.rename(f"{directory}/new_face.jpg", f"{directory}/{new_face_name}.jpg")
        print(f"New face named as {new_face_name}.")
    else:
        os.remove(f"{directory}/new_face.jpg")
        print("New face deleted as no name was provided.")

# check if the detected face is a new face
def is_new_face(face_image, threshold=1000):
    known_faces_folder = 'known_faces'
    min_distance = float('inf')  # Keep track of the closest face match
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            known_face_image = cv2.imread(os.path.join(known_faces_folder, filename))
            known_face_image_gray = to_grayscale(known_face_image)
            known_face_image_resized = cv2.resize(known_face_image_gray, (face_image.shape[1], face_image.shape[0]))
            distance = np.linalg.norm(face_image - known_face_image_resized)
            if distance < min_distance:
                min_distance = distance  # Update if this face is closer
            if distance < threshold:
                return False  # Not a new face
    print(f"Closest face distance: {min_distance}")  # Log closest face distance
    return True  # New face



# start face recognition
def start_recognition():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    # load known faces and their encodings
    known_faces = load_and_encode_images('C:/Users/Solo/Desktop/face_recognition/known_faces')

    frame_number = 0
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_number += 1
            if frame_number % 2 == 0:
                recognize_faces(frame, known_faces)

                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

# create a start GUI
def create_start_gui():
    root = tk.Tk()
    root.title("Face Recognition System")

    start_button = tk.Button(root, text="Start Face Recognition", command=start_recognition)
    start_button.pack()

    root.mainloop()

create_start_gui()

