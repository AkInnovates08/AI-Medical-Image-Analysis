🧠 AI-Powered Medical Image Analysis System
---
📌 Overview

This project is an AI-based system that detects Pneumonia from Chest X-ray images using Deep Learning.
It also includes a Streamlit web application where users can upload an X-ray image and get instant predictions.
---
🚀 Features

📊 Deep Learning model (MobileNetV2 - Transfer Learning)

🏥 Pneumonia detection from X-ray images

🌐 Streamlit web app (real-time prediction)

⚡ Fast and lightweight (CPU-friendly)

📈 Confidence score + visual output
---
🏥 Industry Use Case

Hospitals (AI-assisted diagnosis)
Radiology centers
Diagnostic labs
Health-tech startups
---
🧰 Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy
Streamlit
---
📂 Dataset

Chest X-ray Pneumonia Dataset (Kaggle)

Classes: NORMAL, PNEUMONIA
Real medical imaging dataset
---
🏗️ Project Structure
---
AI-Medical-Image-Analysis/
│
├── data/
├── models/
├── src/
├── app/
├── outputs/
├── images/
├── README.md
├── requirements.txt
---
pip install -r requirements.txt
---
▶️ Run Training
python src/train.py
---
🌐 Run Web App
streamlit run app/app.py
---
📊 Results
Training Accuracy: ~98%
Validation Accuracy: ~75–85%

---
🧠 How It Works
Upload X-ray image
Image preprocessing (resize, normalize)
Deep Learning model prediction
Output: Pneumonia / Normal
---
⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.
