import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from models import MediumCustomCNN,CustomCNN
# Load YuNet model (ONNX)
modelFile = "face_detection_model.onnx"
if not os.path.isfile(modelFile):
    raise FileNotFoundError(f"{modelFile} not found! Please download it manually and place it in the project folder.")

# Create YuNet face detector
detector = cv2.FaceDetectorYN.create(
    model=modelFile,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = MediumCustomCNN()
emotion_model.load_state_dict(torch.load("best_model_emotions.pth", map_location=device))
glasses_model = CustomCNN()
glasses_model.load_state_dict(torch.load("best_model_glasses.pth", map_location=device))
emotion_model.eval()
glasses_model.eval()


# Transform for emotion model (1-channel grayscale)
emotion_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transform for glasses model (3-channel RGB)
glasses_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_face_orientation(image):
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, results = detector.detect(image)

    orientation = "No face detected"
    cropped_face = None  # To hold the cropped face image

    if results is not None and len(results) > 0:
        face = results[0]
        x, y, w_face, h_face = face[:4]
        landmarks = face[4:14].reshape((5, 2))
        score = face[14]

        startX, startY = int(x), int(y)
        endX, endY = int(x + w_face), int(y + h_face)

        # Make bounding box a bit bigger
        buffer = 20  # Buffer to increase bounding box size
        startX = max(startX - buffer, 0)
        startY = max(startY - buffer, 0)
        endX = min(endX + buffer, w)
        endY = min(endY + buffer, h)

        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)

        l_eye_x, l_eye_y = landmarks[1]  # left eye
        r_eye_x, r_eye_y = landmarks[0]  # right eye
        nose_x, nose_y = landmarks[2]

        # Logic based on nose position relative to the left and right boundaries
        if nose_x < r_eye_x:  # Nose is to the left of the face box
            orientation = "Right Profile"
        elif nose_x > l_eye_x:  # Nose is to the right of the face box
            orientation = "Left Profile"
        else:
            orientation = "Frontal Face"

        cv2.putText(image, orientation, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Crop the face region
        cropped_face = image[startY:endY, startX:endX]

    return orientation, image, cropped_face

def classify_face(cropped_face):
    if cropped_face is not None:
        # Convert to PIL for processing
        cropped_face_pil_rgb = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

        # Prepare input for glasses model (RGB)
        glasses_input = glasses_transform(cropped_face_pil_rgb).unsqueeze(0).to(device)

        # Prepare input for emotion model (Grayscale)
        cropped_face_pil_gray = cropped_face_pil_rgb.convert("L")
        emotion_input = emotion_transform(cropped_face_pil_gray).unsqueeze(0).to(device)

        with torch.no_grad():
            emotion_output = emotion_model(emotion_input)
            glasses_output = glasses_model(glasses_input)

            emotion_prediction = torch.argmax(emotion_output, dim=1).item()
            glasses_prediction = torch.argmax(glasses_output, dim=1).item()

        return emotion_prediction, glasses_prediction

    return None, None


# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
glasses_labels = ['No Glasses', 'Glasses']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    orientation, output_image, cropped_face = detect_face_orientation(frame)

    emotion_prediction, glasses_prediction = classify_face(cropped_face)

    # Overlay predictions
    y_offset = 40
    if emotion_prediction is not None:
        cv2.putText(output_image, f"Emotion: {emotion_labels[emotion_prediction]}", 
                    (10, output_image.shape[0] - y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset -= 30

    if glasses_prediction is not None:
        cv2.putText(output_image, f"Glasses: {glasses_labels[glasses_prediction]}", 
                    (10, output_image.shape[0] - y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(output_image, f"Orientation: {orientation}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Orientation and Classification", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
