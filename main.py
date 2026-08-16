import cv2
import numpy as np
import os
import sys

# Suppress TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Error: TensorFlow is not installed. Please run: pip install tensorflow")
    sys.exit(1)

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

def load_ai_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Error: Model files missing!")
        print(f"Please place '{MODEL_PATH}' and '{LABELS_PATH}' in the current directory.")
        print("Export them from Google Teachable Machine (TensorFlow -> Keras format).")
        sys.exit(1)

    print("Loading AI model...")
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model and labels loaded successfully!")
    return model, class_names

def main():
    model, class_names = load_ai_model()

    # Initialize webcam (0 is default integrated camera)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not access webcam. Please check permissions or device index.")
        sys.exit(1)

    print("\n==================================================")
    print("  EcoScan AI - Real-time Waste Sorter Active")
    print("  Point an object at the camera.")
    print("  Press [ESC] to exit.")
    print("==================================================\n")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture video frame. Exiting...")
            break

        # Resize image for Teachable Machine input requirement (224x224 pixels)
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert image frame to numpy array and reshape for model input
        image_array = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
        
        # Normalize pixel values (-1 to +1 range expected by Keras model)
        normalized_image = (image_array / 127.5) - 1.0

        # Perform AI inference prediction
        prediction = model.predict(normalized_image, verbose=0)
        index = np.argmax(prediction)
        
        # Extract class label (cleaning up any index prefix like "0 Paper")
        raw_label = class_names[index]
        label_parts = raw_label.split(' ', 1)
        class_name = label_parts[1] if len(label_parts) > 1 else raw_label
        
        confidence_score = float(prediction[0][index])

        # Draw UI overlay on video frame
        height, width, _ = frame.shape
        
        # Semi-transparent overlay bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (30, 30, 30), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if confidence_score > 0.70:
            text = f"Detected: {class_name} ({int(confidence_score * 100)}%)"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Scanning for item...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

        # Show main video window
        cv2.imshow("EcoScan AI - Waste Classification", frame)

        # Check for ESC key press (ASCII code 27)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting application...")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
