from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import os
PORT = int(os.environ.get("PORT", 10000))
app = FastAPI()

# ----------------------------
# Load models once (IMPORTANT)
# ----------------------------
detector = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings
import os

if os.path.exists("face_embeddings.pkl"):
    with open("face_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    X = data["embeddings"]
    y = data["labels"]
else:
    # 🔥 create empty database
    X = np.empty((0, 512))
    y = np.array([])

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')

    # Avoid division by zero
    std = face_pixels.std()
    if std == 0:
        std = 1

    face_pixels = (face_pixels - face_pixels.mean()) / std

    face_tensor = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding[0].numpy()


def recognize_face(face, threshold=0.5):
    embedding = get_embedding(face)

    similarities = cosine_similarity([embedding], X)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        label = y[best_idx]
        return label, float(best_score)
    else:
        return "Unknown", float(best_score)


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def home():
    return {"message": "Face Recognition API is running"}


# ----------------------------
# PREDICT
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb)

    if boxes is None:
        return {"num_faces": 0, "results": []}

    output = []

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        face = rgb[y1:y2, x1:x2]

        try:
            face = cv2.resize(face, (160, 160))
        except:
            continue

        name, score = recognize_face(face)

        output.append({
            "name": name,
            "confidence": score,
            "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        })

    return {
        "num_faces": len(output),
        "results": output
    }


# ----------------------------
# ADD PERSON
# ----------------------------
@app.post("/add-person")
async def add_person(name: str = Form(...), file: UploadFile = File(...)):
    global X, y

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb)

    if boxes is None:
        return {"error": "No face detected"}

    # Take first face
    x1, y1, x2, y2 = boxes[0].astype(int)

    face = rgb[y1:y2, x1:x2]

    try:
        face = cv2.resize(face, (160, 160))
    except:
        return {"error": "Face resize failed"}

    embedding = get_embedding(face)

    # Append data
    X = np.vstack([X, embedding])
    y = np.append(y, name)

    # Save updated embeddings
    with open("face_embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": X, "labels": y}, f)

    unique_people = sorted(list(set(y)))

    return {
        "message": f"{name} added successfully",
        "total_people": len(unique_people),
        "people": unique_people
    }


# ----------------------------
# DELETE PERSON
# ----------------------------
@app.delete("/delete-person")
async def delete_person(name: str = Form(...)):
    global X, y

    if name not in y:
        return {"error": f"{name} not found"}

    mask = y != name

    X = X[mask]
    y = y[mask]

    with open("face_embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": X, "labels": y}, f)

    unique_people = sorted(list(set(y)))

    return {
        "message": f"{name} deleted successfully",
        "total_people": len(unique_people),
        "people": unique_people
    }


# ----------------------------
# LIST PEOPLE
# ----------------------------
@app.get("/list-people")
def list_people():
    unique_people = sorted(list(set(y)))
    return {"people": unique_people}