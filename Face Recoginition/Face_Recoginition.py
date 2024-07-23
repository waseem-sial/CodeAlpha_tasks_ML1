import cv2
import numpy as np
import tensorflow as tf

# Load a pre-trained face detection model
def load_face_detection_model():
    # Load the pre-trained model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Detect faces in an image
def detect_faces(image, face_cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Draw rectangles around detected faces
def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    face_cascade = load_face_detection_model()
    faces = detect_faces(image, face_cascade)
    image_with_faces = draw_faces(image, faces)
    return image_with_faces

# Save the result
def save_result(image, output_path):
    cv2.imwrite(output_path, image)

# Main function
if __name__ == "__main__":
    image_path = 'D:\\CodeAlpha_tasks\\Face Recoginition\\test_image.jpg'
    output_path = 'D:\\CodeAlpha_tasks\\Face Recoginition\\output_image.jpg'
    
    # Process the image
    processed_image = preprocess_image(image_path)
    
    # Save the processed image
    save_result(processed_image, output_path)
    
    # Display the result
    cv2.imshow('Face Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
