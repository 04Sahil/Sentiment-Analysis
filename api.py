from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2
from student_affect_monitor import EAR_THRESH, harmonise, mp_face_mesh, LEFT_EYE, RIGHT_EYE
from deepface import DeepFace
from scipy.spatial import distance as dist

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def ear(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

@app.post("/affect")
async def affect(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "upload an image")
    img = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "bad image")

    # Check tired
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    tired = False
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        l_eye = [(lm[i].x, lm[i].y) for i in LEFT_EYE]
        r_eye = [(lm[i].x, lm[i].y) for i in RIGHT_EYE]
        if ear(l_eye + r_eye) < EAR_THRESH:
            tired = True

    # DeepFace emotion
    try:
        out = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        raw = out[0]["dominant_emotion"] if isinstance(out, list) else out["dominant_emotion"]
        score = out[0]["emotion"][raw] if isinstance(out, list) else out["emotion"][raw]
    except Exception:
        raw, score = "neutral", 0.0

    label = harmonise(raw, tired)
    return {"label": label, "score": float(score)}
