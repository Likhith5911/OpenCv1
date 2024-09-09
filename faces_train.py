import os
import cv2 as cv
import numpy as np

# Define the list of people (celebrities) to recognize
people = ['Shah Rukh', 'Adam Sandler', 'Kevin Jart']

# Directory where the images are stored
DIR = r'C:\Users\likhith\Downloads\OpenFace'

# Load the Haar cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Lists to hold the features (face regions) and labels (person identifiers)
features = []
labels = []

# Function to create the training data
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        # Check if the directory exists
        if not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            continue

        # Iterate through each image in the person's folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Read the image
            img_array = cv.imread(img_path)
            if img_array is None:
                continue  # Skip if the image can't be loaded

            # Convert the image to grayscale
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # For each detected face, extract the region of interest (ROI)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

# Create training data
create_train()

# Convert the features and labels to NumPy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Initialize the Local Binary Patterns Histograms (LBPH) face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features and labels
face_recognizer.train(features, labels)

# Save the trained model and the features/labels
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Training complete. Model, features, and labels saved.")

# Now, perform face recognition on a new image
def recognize_face(img_path):
    # Load the trained model
    face_recognizer.read('face_trained.yml')

    # Load an image for face recognition
    img = cv.imread(img_path)

    # Check if image is loaded correctly
    if img is None:
        print(f"Failed to load image at {img_path}")
        return

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Show the grayscale image
    cv.imshow('Grayscale Image', gray)

    # Detect faces in the image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        # Recognize the face using the trained model
        label, confidence = face_recognizer.predict(faces_roi)
        
        # Output confidence for debugging
        print(f'Face detected: Label = {people[label]} with a confidence of {confidence}')

        # Annotate the image with the label and draw a rectangle around the face
        cv.putText(img, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Show the detected face(s) with labels
    cv.imshow('Detected Face', img)

    # Wait for a key press before closing the windows
    cv.waitKey(0)
    cv.destroyAllWindows()

# Path to the test image
test_img_path = r'C:\Users\likhith\Downloads\OpenFace\Kevin Jart\Screenshot 2024-09-07 025423.png'
recognize_face(test_img_path)
