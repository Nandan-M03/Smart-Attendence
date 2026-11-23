"""Smart Attendance AI face service (InsightFace buffalo_l). Serves the app and the /api face endpoints.
Templates are stored encrypted on disk and are never returned by any endpoint."""
import base64, json, os, threading
from pathlib import Path
import cv2, numpy as np
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from insightface.app import FaceAnalysis

MATCH = float(os.getenv("MATCH_THRESHOLD", "0.45"))        # cosine similarity needed to match
YAW_SPREAD = float(os.getenv("LIVENESS_YAW_SPREAD", "20"))  # degrees of head turn required
STORE = Path(os.getenv("FACE_STORE", "data/faces.enc"))
KEY = os.environ.get("FACE_KEY")
if not KEY:
    raise SystemExit('Set FACE_KEY first: python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"')
box, lock = Fernet(KEY), threading.Lock()
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(960, 960))
app = FastAPI()

def load():
    return json.loads(box.decrypt(STORE.read_bytes())) if STORE.exists() else {}

def save(d):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_bytes(box.encrypt(json.dumps(d).encode()))

def img(data):
    try:
        im = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",", 1)[-1]), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        im = None
    if im is None:
        raise HTTPException(400, "Invalid image")
    return im

class Enroll(BaseModel): id: str; image: str
class Verify(BaseModel): id: str; frames: list[str]
class Recog(BaseModel): ids: list[str]; image: str

@app.get("/api/health")
def health():
    return {"engine": True}

@app.post("/api/enroll")
def enroll(b: Enroll):
    faces = fa.get(img(b.image))
    if len(faces) != 1:
        raise HTTPException(422, f"The photo must contain exactly one face (found {len(faces)})")
    if faces[0].det_score < 0.6:
        raise HTTPException(422, "Face is not clear enough. Use a front-facing, well-lit photo.")
    with lock:
        d = load(); d[b.id] = faces[0].normed_embedding.tolist(); save(d)
    return {"ok": True}

@app.post("/api/verify-teacher")
def verify(b: Verify):
    ref = load().get(b.id)
    if ref is None:
        raise HTTPException(404, "This teacher has no enrolled face")
    if not 5 <= len(b.frames) <= 20:
        raise HTTPException(400, "Send 5 to 20 frames")
    ref, yaws = np.array(ref), []
    for f in b.frames:
        faces = fa.get(img(f))
        if len(faces) != 1:
            continue
        if float(ref @ faces[0].normed_embedding) < MATCH:
            return {"verified": False, "reason": "Face does not match the enrolled teacher."}
        pose = getattr(faces[0], "pose", None)
        if pose is None:
            raise HTTPException(500, "Pose model missing; use the buffalo_l model pack")
        yaws.append(float(pose[1]))
    if len(yaws) < 4:
        return {"verified": False, "reason": "Your face was not clearly visible. Try again."}
    if max(yaws) - min(yaws) < YAW_SPREAD:
        return {"verified": False, "reason": "Liveness check failed. Turn your head slowly left and right."}
    return {"verified": True}

@app.post("/api/recognize")
def recognize(b: Recog):
    d = load()
    roster = [(i, np.array(d[i])) for i in b.ids if i in d]
    faces, best = fa.get(img(b.image)), {}
    for f in faces:
        sc = [(float(e @ f.normed_embedding), i) for i, e in roster]
        if sc:
            s, i = max(sc)
            if s >= MATCH and s > best.get(i, 0):
                best[i] = s
    return {"faces": len(faces), "matches": [{"id": i, "score": round(s, 3)} for i, s in best.items()]}

@app.delete("/api/face/{uid}")
def forget(uid: str):
    with lock:
        d = load(); d.pop(uid, None); save(d)
    return {"ok": True}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
