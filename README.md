# EcoScan AI ♻️

> An edge-AI camera application built with Python and TensorFlow that identifies waste types in real time using computer vision. Designed as an introductory project for Grade 7 coding & artificial intelligence learning.

---

## 📌 Project Overview
**EcoScan AI** turns any laptop webcam into a smart waste-sorting camera. Using a custom-trained TensorFlow / Keras model and OpenCV, the application analyzes camera frames in real-time to categorize objects into **Compost**, **Recycling**, or **Landfill Trash**. 

This project runs **100% locally and offline**—requiring no cloud APIs, user accounts, or internet access once configured.

---

## 🚀 Features
* 🎥 **Real-Time Detection:** Live webcam overlay showing predicted waste category and confidence score.
* 🔒 **Private & Local:** Complete on-device processing via TensorFlow and OpenCV.
* 🎯 **Confidence Filtering:** Ignores predictions under 70% certainty to minimize false positives.
* ♻️ **Interactive Hints:** Real-time feedback on how to properly dispose of detected items.
* 📶 **Zero External Dependencies:** No cloud registration, API keys, or active Wi-Fi required.

---

## 🛠️ Requirements & Tech Stack
* **Language:** Python 3.8+
* **Libraries:** `opencv-python`, `tensorflow`, `numpy`
* **Model:** Pre-trained Keras model (`keras_Model.h5`) + class labels (`labels.txt`)

---

## 📥 Installation

1. **Clone or Download** this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/ecoscan-ai.git
   cd ecoscan-ai
   ```

2. **Install Required Python Packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎓 Model Training (Teachable Machine)

To train or customize your AI waste sorter model:

1. Open **[Teachable Machine](https://teachablemachine.withgoogle.com/)** in your web browser (no account needed).
2. Click **Get Started** → **Image Project** → **Standard Image Model**.
3. Create classes (e.g., `Paper`, `Plastic`, `Compost`, `Trash`).
4. Use your webcam to record ~30–50 photos for each class from various angles.
5. Click **Train Model** and wait for training to complete.
6. Click **Export Model** → Select **TensorFlow** tab → Choose **Keras**.
7. Click **Download my model** to get the `.zip` archive.
8. Extract `keras_Model.h5` and `labels.txt` into the root folder of this repository.

---

## 🚀 Usage

Run the main application script:

```bash
python main.py
```

* **Live Feed:** Hold objects in front of your camera to see classification results.
* **Exit:** Press the `ESC` key to safely shut down the video stream.

---

## 📂 Project Structure

```
ecoscan-ai/
├── main.py              # Main Python script for camera capture & model inference
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Ignored files (e.g., model weights, bytecode)
├── LICENSE              # MIT Open Source License
├── keras_Model.h5       # (User-provided) Trained TensorFlow model
└── labels.txt           # (User-provided) Class labels file
```

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **`Camera error: Could not open webcam`** | Check webcam privacy settings or change `cv2.VideoCapture(0)` to index `1` or `2`. |
| **`FileNotFoundError: keras_Model.h5`** | Ensure model files exported from Teachable Machine are placed in the same folder as `main.py`. |
| **`ModuleNotFoundError`** | Run `pip install -r requirements.txt` in your active Python environment. |

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.
