import face_recognition
import cv2
import numpy as np
import os

def load_and_encode_images(image_folder):
    face_encodings = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_name = os.path.splitext(filename)[0]
                face_encodings[face_name] = encodings[0]
            else:
                print(f"No faces found in the image {filename}.")
    return face_encodings

def save_new_face(image, name, directory='known_faces'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"{directory}/{name}.jpg", image)

known_faces_folder = 'known_faces'
known_faces = load_and_encode_images(known_faces_folder)

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Could not open video device")

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

        rgb_frame = frame[:, :, ::-1]

        # Only process every other frame to save time
        if frame_number % 2 == 0:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
                best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None
                if best_match_index is not None and matches[best_match_index]:
                    name = list(known_faces.keys())[best_match_index]
                else:
                    face_id = f"new_face_{len(new_faces) + 1}"
                    face_trackers[face_id] = {'count': 1, 'face_image': rgb_frame[face_location[0]:face_location[2], face_location[3]:face_location[1]]}
                    name = face_id

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
        if face_info['count'] > 10:  # Confirm face only if seen more than 10 times
            cv2.imshow("Save Face?", face_info['face_image'])
            cv2.waitKey(1)
            user_input = input(f"Enter name for {face_id} or press enter to skip: ").strip()
            if user_input:
                save_new_face(face_info['face_image'], user_input)
            cv2.destroyAllWindows()
