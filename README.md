# Human Face Analysis Project

This project is a real-time face analysis system using a webcam. It detects faces, classifies their orientation (frontal, left profile, or right profile), detects whether the person is wearing glasses, and predicts their facial emotion.

---

## Features

* 📸 **Face Detection**: Uses YuNet (ONNX model) for real-time face detection with landmark detection.
* 🔄 **Face Orientation Detection**: Determines if the detected face is frontal, left profile, or right profile based on nose and eye landmarks.
* 🧠 **Emotion Recognition**: Classifies facial emotions using a grayscale CNN model.
* 👓 **Glasses Detection**: Classifies if a person is wearing glasses using a color CNN model.

---

## Requirements

* Python 3.x
* PyTorch
* OpenCV (with `opencv-contrib-python` for `FaceDetectorYN`)
* torchvision
* Pillow

---

## Files

* `face_detection_model.onnx` - Pre-trained YuNet model for face detection
* `best_model_emotions.pth` - Trained PyTorch model for emotion recognition
* `best_model_glasses.pth` - Trained PyTorch model for glasses detection
* `models.py` - Contains definitions for `MediumCustomCNN` and `CustomCNN`

---

## How it Works

1. **Video Capture**: Opens the webcam stream.
2. **Face Detection**: Uses YuNet to detect faces and facial landmarks.
3. **Orientation Estimation**: Uses relative nose and eye positions to classify orientation.
4. **Face Cropping**: Extracts the face region for classification.
5. **Emotion & Glasses Classification**:

   * Converts to grayscale for emotion classification
   * Uses RGB for glasses classification
6. **Display**: Annotates the frame with orientation, emotion, and glasses prediction.

---

## Usage

1. Place all required model files in the project folder:

   * `face_detection_model.onnx`
   * `best_model_emotions.pth`
   * `best_model_glasses.pth`

2. Run the script:

   ```bash
   python app.py
   ```

3. The webcam window will open showing live face detection with:

   * Orientation
   * Emotion (e.g., Happy, Sad, Angry, etc.)
   * Glasses (Yes/No)

4. Press `q` to quit the application.

---

## Emotion Classes

* Angry
* Disgusted
* Fearful
* Happy
* Neutral
* Sad
* Surprised

## Glasses Classes

* No Glasses
* Glasses

---

## Sample Output

Annotated webcam feed showing:

* Detected face bounding box
* Orientation label (e.g., "Frontal Face")
* Emotion prediction (e.g., "Happy")
* Glasses prediction (e.g., "No Glasses")

---

## Notes

* Ensure good lighting for best accuracy.
* The performance may vary based on the quality of your webcam and model training.
* Orientation logic is based on relative nose-eye positions and works best on frontal or slightly angled faces.

---

## Credits

* Face detection by [YuNet (ONNX)](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
* Emotion & Glasses classifiers trained using CNN models with PyTorch

---

## License

This project is for educational and research purposes.
