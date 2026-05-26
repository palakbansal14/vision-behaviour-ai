import os, time, tempfile, collections
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch, cv2, gradio as gr
import plotly.graph_objects as go
from ultralytics import YOLO
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
YUNET    = os.path.expanduser("~/.cv2_models/face_detection_yunet_2023mar.onnx")

yolo          = YOLO("yolov8n-pose.pt")
emotion_model = HSEmotionRecognizer(model_name="enet_b2_8")

EMOTIONS = ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
POSES    = ["Standing","Sitting","Arms Raised","Bending","Leaning"]

CV_COLORS = {
    "Anger":(0,0,220),"Contempt":(0,140,140),"Disgust":(0,120,120),
    "Fear":(160,0,160),"Happiness":(0,200,0),"Neutral":(160,160,160),
    "Sadness":(200,60,60),"Surprise":(0,200,220),
}
HEX_EMO = {
    "Anger":"#ef4444","Contempt":"#06b6d4","Disgust":"#14b8a6",
    "Fear":"#a855f7","Happiness":"#22c55e","Neutral":"#64748b",
    "Sadness":"#f97316","Surprise":"#3b82f6",
}
HEX_POSE = {
    "Standing":"#22c55e","Sitting":"#fbbf24","Arms Raised":"#3b82f6",
    "Bending":"#f97316","Leaning":"#a855f7",
}
POSE_CV = {
    "Standing":(0,255,128),"Sitting":(255,200,0),"Arms Raised":(0,180,255),
    "Bending":(255,100,0),"Leaning":(200,0,200),
}

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
KPT_THRESH = 0.3


# ── keypoint helpers ──────────────────────────────────────────────────────────
def _pt(kpts, idx):
    if kpts is None or idx >= len(kpts):
        return None
    x, y, c = kpts[idx]
    return (float(x), float(y)) if float(c) > KPT_THRESH else None

def _conf(kpts, idx):
    return float(kpts[idx][2]) if kpts is not None and idx < len(kpts) else 0.0


# ── face helpers ──────────────────────────────────────────────────────────────
def make_face_det():
    return cv2.FaceDetectorYN.create(
        YUNET, "", (320, 320), score_threshold=0.5, nms_threshold=0.3)

def get_face(det, crop):
    h, w = crop.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(crop)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[-1])
    x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    p = 6
    x1, y1 = max(0, x-p), max(0, y-p)
    x2, y2 = min(w, x+fw+p), min(h, y+fh+p)
    if x2-x1 < 20 or y2-y1 < 20:
        return None
    face = crop[y1:y2, x1:x2]
    fh2, fw2 = face.shape[:2]
    if fh2 < 96 or fw2 < 96:
        s = max(96/fh2, 96/fw2)
        face = cv2.resize(face, (int(fw2*s), int(fh2*s)))
    return face


# ── pose classifier ───────────────────────────────────────────────────────────
def classify_pose(kpts):
    l_sh,  r_sh  = _pt(kpts, 5),  _pt(kpts, 6)
    l_wr,  r_wr  = _pt(kpts, 9),  _pt(kpts, 10)
    l_hip, r_hip = _pt(kpts, 11), _pt(kpts, 12)
    l_kn,  r_kn  = _pt(kpts, 13), _pt(kpts, 14)

    if l_wr and l_sh and l_wr[1] < l_sh[1] - 15:
        return "Arms Raised"
    if r_wr and r_sh and r_wr[1] < r_sh[1] - 15:
        return "Arms Raised"

    if l_sh and r_sh and l_hip and r_hip:
        sh_y  = (l_sh[1]  + r_sh[1])  / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        torso = hip_y - sh_y
        if torso < 0:
            return "Bending"
        if l_kn and r_kn and torso > 0:
            kn_y = (l_kn[1] + r_kn[1]) / 2
            if (kn_y - hip_y) < torso * 0.4:
                return "Sitting"
        tilt  = abs(l_sh[1] - r_sh[1])
        width = abs(l_sh[0] - r_sh[0])
        if width > 0 and tilt / width > 0.35:
            return "Leaning"

    return "Standing"


# ── head direction ────────────────────────────────────────────────────────────
def detect_head_direction(kpts):
    nose  = _pt(kpts, 0)
    l_eye = _pt(kpts, 1)
    r_eye = _pt(kpts, 2)
    l_ear_c = _conf(kpts, 3)
    r_ear_c = _conf(kpts, 4)

    if nose and l_eye and r_eye:
        eye_y  = (l_eye[1] + r_eye[1]) / 2
        eye_dx = abs(l_eye[0] - r_eye[0])
        if nose[1] > eye_y + max(eye_dx * 0.55, 12):
            return "Head Down"

    if r_ear_c < 0.2 and l_ear_c > 0.3:
        return "Turned Left"
    if l_ear_c < 0.2 and r_ear_c > 0.3:
        return "Turned Right"

    return "Fwd"


# ── hand positions ────────────────────────────────────────────────────────────
def detect_hand_positions(kpts):
    l_sh,  r_sh  = _pt(kpts, 5),  _pt(kpts, 6)
    l_wr,  r_wr  = _pt(kpts, 9),  _pt(kpts, 10)
    l_hip, r_hip = _pt(kpts, 11), _pt(kpts, 12)
    l_ank, r_ank = _pt(kpts, 15), _pt(kpts, 16)
    l_wr_c = _conf(kpts, 9)
    r_wr_c = _conf(kpts, 10)

    body_cx = (l_hip[0] + r_hip[0]) / 2 if l_hip and r_hip else None

    def classify(wrist, wrist_conf, shoulder, hip, ankle, is_left):
        if wrist_conf < 0.15:
            return "Behind"
        if wrist is None:
            return "Hidden"
        wx, wy = wrist
        if shoulder and wy < shoulder[1] - 15:
            return "Raised"
        if ankle and wy > ankle[1] - 70:
            return "Near Feet"
        if hip and abs(wy - hip[1]) < 50 and abs(wx - hip[0]) < 65:
            return "Pocket"
        if body_cx is not None:
            if is_left  and wx > body_cx + 30: return "Crossed"
            if not is_left and wx < body_cx - 30: return "Crossed"
        if hip and wy > hip[1] + 25:
            return "Down"
        return "Mid"

    lp = classify(l_wr, l_wr_c, l_sh, l_hip, l_ank, True)
    rp = classify(r_wr, r_wr_c, r_sh, r_hip, r_ank, False)
    return lp, rp


# ── leg stance ────────────────────────────────────────────────────────────────
def detect_leg_stance(kpts):
    l_hip, r_hip = _pt(kpts, 11), _pt(kpts, 12)
    l_ank, r_ank = _pt(kpts, 15), _pt(kpts, 16)

    if l_ank and r_ank:
        dy = abs(l_ank[1] - r_ank[1])
        if dy > 80:
            return "L-Leg Up" if l_ank[1] < r_ank[1] else "R-Leg Up"
        ankle_dx = abs(l_ank[0] - r_ank[0])
        if l_hip and r_hip:
            hip_w = max(abs(l_hip[0] - r_hip[0]), 1)
            ratio = ankle_dx / hip_w
            if ratio > 1.8: return "Wide Stance"
            if ratio < 0.3: return "Feet Together"

    return "Normal"


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_skeleton(frame, kpts):
    if kpts is None:
        return
    pts = [(int(x), int(y), float(c)) for x, y, c in kpts]
    for i, j in SKELETON:
        if i < len(pts) and j < len(pts):
            if pts[i][2] > KPT_THRESH and pts[j][2] > KPT_THRESH:
                cv2.line(frame, pts[i][:2], pts[j][:2], (0, 255, 128), 2)
    for x, y, c in pts:
        if c > KPT_THRESH:
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)


def _tag(frame, text, x, y, bg, text_color=(0,0,0), font_scale=0.45, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    by = y + th + 5
    cv2.rectangle(frame, (x, y), (x+tw+6, by), bg, -1)
    cv2.putText(frame, text, (x+3, by-3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return by


def draw_box(frame, box, idx, emotion, score, pose, head, l_hand, r_hand, legs):
    x1, y1, x2, y2 = box
    emo_c  = CV_COLORS.get(emotion, (160,160,160)) if emotion else (80,80,80)
    pose_c = POSE_CV.get(pose, (160,160,160))

    cv2.rectangle(frame, (x1,y1), (x2,y2), emo_c, 2)

    lbl = f"#{idx}  {emotion}  {score:.0f}%" if emotion else f"#{idx}"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    ly = max(y1, th+6)
    cv2.rectangle(frame, (x1, ly-th-6), (x1+tw+6, ly), emo_c, -1)
    cv2.putText(frame, lbl, (x1+3, ly-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2)

    cy = y2 + 2
    cy = _tag(frame, pose,                         x1, cy, pose_c)
    cy = _tag(frame, f"Head : {head}",             x1, cy, (30,30,50), (220,220,220))
    cy = _tag(frame, f"LH:{l_hand}  RH:{r_hand}", x1, cy, (30,30,50), (220,220,220))
    cy = _tag(frame, f"Legs : {legs}",             x1, cy, (30,30,50), (220,220,220))


def plot_bg():
    return dict(paper_bgcolor="#111118", plot_bgcolor="#111118",
                font=dict(color="#94a3b8", size=12),
                margin=dict(t=48, b=36, l=36, r=16))


# ── processing ───────────────────────────────────────────────────────────────
def process_video(video_path, progress=gr.Progress()):
    conf_thresh, skip_n = 0.35, 3
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    face_det = make_face_det()
    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    emotion_log    = []
    pose_log       = []
    emotion_totals = collections.Counter()
    pose_totals    = collections.Counter()
    max_people     = 0
    cached         = {}
    prev_boxes     = []
    prev_kpts      = []
    t0 = time.time()

    progress(0, desc="Starting…")
    for fi in range(total):
        ret, frame = cap.read()
        if not ret: break

        out = yolo(frame, classes=[0], verbose=False, device=DEVICE, conf=float(conf_thresh))
        boxes, kpts_list = [], []
        if out[0].boxes is not None:
            for i, b in enumerate(out[0].boxes):
                boxes.append(tuple(map(int, b.xyxy[0])))
                if out[0].keypoints is not None and i < len(out[0].keypoints.data):
                    kpts_list.append(out[0].keypoints.data[i].cpu().numpy())
                else:
                    kpts_list.append(None)
        prev_boxes = boxes
        prev_kpts  = kpts_list
        max_people = max(max_people, len(boxes))

        if fi % int(skip_n) == 0:
            nc = {}
            for i, (x1,y1,x2,y2) in enumerate(boxes):
                kpts = prev_kpts[i] if i < len(prev_kpts) else None

                pose   = classify_pose(kpts)
                head   = detect_head_direction(kpts)
                l_hand, r_hand = detect_hand_positions(kpts)
                legs   = detect_leg_stance(kpts)

                pose_log.append({"f": fi, "p": pose})
                pose_totals[pose] += 1

                crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                if crop.size == 0:
                    nc[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)
                    continue

                face = get_face(face_det, crop)
                if face is None:
                    nc[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)
                    continue

                try:
                    emo, scores = emotion_model.predict_emotions(face, logits=False)
                    sc = float(max(scores)) * 100
                    nc[i] = (emo, sc, pose, head, l_hand, r_hand, legs, kpts)
                    emotion_log.append({"f": fi, "e": emo, "s": sc})
                    emotion_totals[emo] += 1
                except Exception:
                    nc[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)
            cached = nc

        annotated = frame.copy()
        for i, box in enumerate(prev_boxes):
            emo, sc, pose, head, l_hand, r_hand, legs, kpts = cached.get(
                i, (None, 0, "Standing", "Fwd", "Mid", "Mid", "Normal", None))
            draw_skeleton(annotated, kpts)
            draw_box(annotated, box, i+1, emo, sc, pose, head, l_hand, r_hand, legs)

        ts = fi / fps
        cv2.putText(annotated, f"{int(ts//60):02d}:{int(ts%60):02d}  |  {len(prev_boxes)} people",
                    (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        writer.write(annotated)

        if fi % 20 == 0:
            pct = fi / total
            eta = (time.time()-t0) / max(pct, 0.01) * (1-pct)
            progress(pct, desc=f"Frame {fi}/{total}  —  ETA {eta:.0f}s")

    cap.release(); writer.release()

    # Re-encode to H.264 so browsers can play it in Gradio
    final_path = tempfile.mktemp(suffix=".mp4")
    os.system(
        f'ffmpeg -y -i "{out_path}" '
        f'-vcodec libx264 -preset fast -crf 23 '
        f'-movflags +faststart -an "{final_path}" '
        f'-loglevel error'
    )
    os.remove(out_path)

    # ── Emotion Donut ────────────────────────────────────────────────────
    if emotion_totals:
        el = list(emotion_totals.keys())
        ev = list(emotion_totals.values())
        emo_pie = go.Figure(go.Pie(
            labels=el, values=ev,
            marker=dict(colors=[HEX_EMO.get(e,"#888") for e in el],
                        line=dict(color="#111118", width=2)),
            hole=0.6, textinfo="label+percent",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>%{value} detections<br>%{percent}<extra></extra>",
        ))
        emo_pie.update_layout(
            **plot_bg(),
            title=dict(text="Emotion Distribution", x=0.5, font=dict(size=14, color="#e2e8f0")),
            showlegend=False,
            annotations=[dict(text=f"<b>{sum(ev)}</b><br><span style='font-size:10px'>detections</span>",
                              x=0.5, y=0.5, showarrow=False, font=dict(color="#e2e8f0", size=14))],
        )
    else:
        emo_pie = go.Figure()
        emo_pie.update_layout(**plot_bg(), title=dict(text="No emotion detections", x=0.5))

    # ── Pose Donut ───────────────────────────────────────────────────────
    if pose_totals:
        pl = list(pose_totals.keys())
        pv = list(pose_totals.values())
        pose_pie = go.Figure(go.Pie(
            labels=pl, values=pv,
            marker=dict(colors=[HEX_POSE.get(p,"#888") for p in pl],
                        line=dict(color="#111118", width=2)),
            hole=0.6, textinfo="label+percent",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>%{value} frames<br>%{percent}<extra></extra>",
        ))
        pose_pie.update_layout(
            **plot_bg(),
            title=dict(text="Pose Distribution", x=0.5, font=dict(size=14, color="#e2e8f0")),
            showlegend=False,
            annotations=[dict(text=f"<b>{sum(pv)}</b><br><span style='font-size:10px'>frames</span>",
                              x=0.5, y=0.5, showarrow=False, font=dict(color="#e2e8f0", size=14))],
        )
    else:
        pose_pie = go.Figure()
        pose_pie.update_layout(**plot_bg(), title=dict(text="No pose data", x=0.5))

    # ── Emotion Timeline ─────────────────────────────────────────────────
    if emotion_log:
        window  = max(1, int(fps))
        buckets = {}
        for e in emotion_log:
            b = (e["f"] // window) * window
            buckets.setdefault(b, collections.Counter())[e["e"]] += 1
        xs = sorted(buckets)
        emo_timeline = go.Figure()
        for emo in EMOTIONS:
            yv = [buckets[b].get(emo, 0) for b in xs]
            if any(yv):
                emo_timeline.add_trace(go.Scatter(
                    x=[t/fps for t in xs], y=yv, mode="lines", name=emo,
                    line=dict(color=HEX_EMO.get(emo,"#888"), width=2),
                    fill="tozeroy", stackgroup="one",
                    hovertemplate="<b>%{fullData.name}</b><br>%{x:.1f}s  •  %{y}<extra></extra>",
                ))
        emo_timeline.update_layout(
            **plot_bg(),
            title=dict(text="Emotion Timeline", x=0.5, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="seconds", gridcolor="#1e293b", zerolinecolor="#1e293b"),
            yaxis=dict(title="count",   gridcolor="#1e293b", zerolinecolor="#1e293b"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
    else:
        emo_timeline = go.Figure()
        emo_timeline.update_layout(**plot_bg(), title=dict(text="No emotion timeline", x=0.5))

    # ── Pose Timeline ────────────────────────────────────────────────────
    if pose_log:
        window   = max(1, int(fps))
        pbuckets = {}
        for e in pose_log:
            b = (e["f"] // window) * window
            pbuckets.setdefault(b, collections.Counter())[e["p"]] += 1
        xs = sorted(pbuckets)
        pose_timeline = go.Figure()
        for pose in POSES:
            yv = [pbuckets[b].get(pose, 0) for b in xs]
            if any(yv):
                pose_timeline.add_trace(go.Scatter(
                    x=[t/fps for t in xs], y=yv, mode="lines", name=pose,
                    line=dict(color=HEX_POSE.get(pose,"#888"), width=2),
                    fill="tozeroy", stackgroup="one",
                    hovertemplate="<b>%{fullData.name}</b><br>%{x:.1f}s  •  %{y}<extra></extra>",
                ))
        pose_timeline.update_layout(
            **plot_bg(),
            title=dict(text="Pose Timeline", x=0.5, font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(title="seconds", gridcolor="#1e293b", zerolinecolor="#1e293b"),
            yaxis=dict(title="count",   gridcolor="#1e293b", zerolinecolor="#1e293b"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
    else:
        pose_timeline = go.Figure()
        pose_timeline.update_layout(**plot_bg(), title=dict(text="No pose timeline", x=0.5))

    # ── Summary cards ────────────────────────────────────────────────────
    top_emo  = emotion_totals.most_common(1)[0] if emotion_totals else ("—", 0)
    top_pose = pose_totals.most_common(1)[0]    if pose_totals    else ("—", 0)
    dur = int(total / fps)
    cards = f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:14px">
      {"".join(f'''
      <div style="background:#17171f;border:1px solid #23233a;border-radius:12px;
                  padding:16px 12px;text-align:center">
        <div style="font-size:1.3rem;font-weight:700;color:{vc}">{vv}</div>
        <div style="color:#475569;font-size:0.75rem;margin-top:4px">{vl}</div>
      </div>''' for vc,vv,vl in [
          ("#818cf8", f"{dur}s",                         "Duration"),
          ("#34d399", str(max_people),                   "Max People"),
          ("#f472b6", str(sum(emotion_totals.values())), "Detections"),
          ("#fbbf24", top_emo[0],                        "Top Emotion"),
          ("#22c55e", top_pose[0],                       "Top Pose"),
      ])}
    </div>
    """
    return final_path, emo_pie, pose_pie, emo_timeline, pose_timeline, cards


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #0c0c14 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 32px 40px !important;
}
.main { max-width: 100% !important; }
.gap  { max-width: 100% !important; }
#component-0 { max-width: 100% !important; }
footer, .built-with, .svelte-1ipelgc { display: none !important; }
#app-header {
    padding: 40px 0 28px;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 28px;
}
#app-header h1 {
    font-size: 1.75rem; font-weight: 700;
    color: #e2e8f0; margin: 0 0 6px;
    letter-spacing: -0.02em;
}
#app-header p { color: #475569; font-size: 0.88rem; margin: 0; }
.gpu-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: #17171f; border: 1px solid #23233a;
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.75rem; color: #64748b; margin-top: 10px;
}
.gpu-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 6px #22c55e;
}
.panel {
    background: #13131f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
.section-title {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #334155; margin-bottom: 14px;
}
label { color: #94a3b8 !important; font-size: 0.82rem !important; }
input[type=range] { accent-color: #6366f1 !important; }
#run-btn button {
    background: #6366f1 !important; border: none !important;
    color: #fff !important; font-weight: 600 !important;
    font-size: 0.95rem !important; border-radius: 10px !important;
    padding: 13px !important; width: 100% !important;
    cursor: pointer !important; transition: background 0.2s !important;
    margin-top: 8px !important;
}
#run-btn button:hover { background: #4f46e5 !important; }
video { border-radius: 10px !important; }
"""

HEADER = f"""
<div id="app-header">
  <h1>Behaviour &amp; Expression Analyser</h1>
  <p>Detects people · Emotion · Pose · Head direction · Hand positions · Leg stance</p>
  <div class="gpu-tag">
    <div class="gpu-dot"></div>
    {GPU_NAME} &nbsp;·&nbsp; YOLOv8n-Pose + YuNet + EfficientNet-B2
  </div>
</div>
"""

with gr.Blocks(title="Behaviour Analyser", css=CSS) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280, elem_classes="panel"):
            gr.HTML('<div class="section-title">Video Input</div>')
            video_in = gr.Video(label=None, height=240)
            with gr.Row(elem_id="run-btn"):
                run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=2, elem_classes="panel"):
            gr.HTML('<div class="section-title">Output</div>')
            video_out = gr.Video(label=None, height=400)
            stats_out = gr.HTML("")

    gr.HTML('<div style="height:20px"></div>')

    with gr.Row():
        with gr.Column(elem_classes="panel"):
            emo_pie_out = gr.Plot(label=None, show_label=False)
        with gr.Column(elem_classes="panel"):
            pose_pie_out = gr.Plot(label=None, show_label=False)

    gr.HTML('<div style="height:20px"></div>')

    with gr.Row():
        with gr.Column(elem_classes="panel"):
            emo_line_out = gr.Plot(label=None, show_label=False)
        with gr.Column(elem_classes="panel"):
            pose_line_out = gr.Plot(label=None, show_label=False)

    run_btn.click(
        fn=process_video,
        inputs=[video_in],
        outputs=[video_out, emo_pie_out, pose_pie_out, emo_line_out, pose_line_out, stats_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
