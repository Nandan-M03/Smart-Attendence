# Smart Attendance AI with face recognition

Run (Python 3.10+):
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    export FACE_KEY=$(python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())")   # keep this key safe
    uvicorn server:app --port 8000
Open http://localhost:8000 (the first start downloads about 280 MB of models).

Use: Admin > People > Manage > Enroll face (clear, front-facing photo, with written consent).
Teacher > Take attendance > Verify my face (turn head left and right) > Scan classroom > Review > Confirm.

Read before deploying:
- Not yet tested end to end; test with a few real people and tune MATCH_THRESHOLD.
- Liveness is a basic head-turn check; it will not stop a replayed video. Student scans have no liveness check; the teacher review is the safeguard.
- The API has no authentication, and logins in the page are still the browser-side demo. Add real auth (for example Supabase) before exposing this beyond localhost.
- Camera access needs localhost or HTTPS.
- The InsightFace pretrained models are for non-commercial use. For commercial use, license them or swap the three endpoints to a service such as AWS Rekognition or Azure Face.
