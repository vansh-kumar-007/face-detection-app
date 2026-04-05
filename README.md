# 🔍 Face Recognition System (End-to-End Deep Learning Project)

An end-to-end **Face Recognition System** built using Deep Learning, capable of detecting and recognizing faces from images and real-time input. The project includes a full backend API, frontend UI, and cloud deployment.

---

## 🚀 Live Demo

* 🌐 **Frontend (Streamlit UI)**: *[https://face-detection-app-x5aaubnswbi83pbpplxp3v.streamlit.app/]*
* ⚙️ **Backend API (FastAPI on Render)**: *[https://face-detection-app-wgm4.onrender.com/]*
* 📄 **Swagger Docs**: *[https://face-detection-app-wgm4.onrender.com/docs]*

---

## 🧠 Project Overview

This system uses a deep learning model (**FaceNet**) to generate facial embeddings and compare them using similarity metrics to recognize individuals.

---

## ✨ Features

* ✅ Face Detection using MTCNN
* ✅ Face Recognition using FaceNet embeddings
* ✅ Real-time prediction support (local)
* ✅ Add new person (dynamic database)
* ✅ Delete person from database
* ✅ REST API using FastAPI
* ✅ Interactive UI using Streamlit
* ✅ Cloud deployment (Render + Streamlit Cloud)

---

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI Backend → FaceNet Model → Output
```

---

## 🛠️ Tech Stack

* **Deep Learning**: FaceNet (facenet-pytorch)
* **Computer Vision**: OpenCV
* **Backend**: FastAPI
* **Frontend**: Streamlit
* **Deployment**: Render (API), Streamlit Cloud (UI)
* **Language**: Python

---

## 📂 Project Structure

```
FaceDetection/
│
├── api/                # FastAPI backend
│   └── main.py
│
├── webapp/             # Streamlit frontend (local)
│   └── app.py
│
├── face_embeddings.pkl # Stored face embeddings
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Face is detected using MTCNN
2. Face is cropped and resized
3. FaceNet model generates 512-d embedding
4. Embedding is compared with stored embeddings
5. Best match is returned using cosine similarity

---

## 📊 Model Details

* **Model Used**: FaceNet (pretrained on VGGFace2)
* **Embedding Size**: 512
* **Similarity Metric**: Cosine Similarity

---

## 📈 Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 1.0   |
| Precision | 1.0   |
| Recall    | 1.0   |
| F1 Score  | 1.0   |

---

## 🧪 API Endpoints

| Endpoint         | Method | Description                |
| ---------------- | ------ | -------------------------- |
| `/predict`       | POST   | Recognize faces from image |
| `/add-person`    | POST   | Add new person             |
| `/delete-person` | DELETE | Remove person              |
| `/list-people`   | GET    | List all known people      |

---

## 🖥️ Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/vansh-kumar-007/face-detection-app.git
cd face-detection-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

```bash
uvicorn api.main:app --reload
```

### 4. Run Frontend

```bash
streamlit run webapp/app.py
```

---

## ☁️ Deployment

### Backend (Render)

* Deployed using FastAPI
* Handles all model inference

### Frontend (Streamlit Cloud)

* Lightweight UI
* Communicates with Render API

---

## ⚠️ Notes

* Free Render service may sleep after inactivity
* First API call may take a few seconds
* Deployment version uses precomputed embeddings

---

## 🔮 Future Improvements

* Real-time webcam deployment online
* Database integration (MongoDB / PostgreSQL)
* User authentication system
* Face attendance system
* Improved UI/UX

---

## 👨‍💻 Author

**Vansh Kumar**
B.Tech Civil Engineering, DTU

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
