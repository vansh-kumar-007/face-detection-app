import cv2
import pickle
import numpy as np
import torch
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Models
# ----------------------------
detector = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

# ----------------------------
# Load Embeddings
# ----------------------------
with open("face_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

X = data["embeddings"]
y = data["labels"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = (face_pixels - face_pixels.mean()) / face_pixels.std()
    
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
        label = encoder.inverse_transform([y_encoded[best_idx]])[0]
        return label, best_score
    else:
        return "Unknown", best_score


# ----------------------------
# Webcam Start
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = detector.detect_faces(rgb)
    
    for res in results:
        x, y, w, h = res['box']
        x, y = abs(x), abs(y)
        
        face = rgb[y:y+h, x:x+w]
        
        try:
            face = cv2.resize(face, (160, 160))
        except:
            continue
        
        name, score = recognize_face(face)
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Label
        text = f"{name} ({score:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()