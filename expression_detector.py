import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import cv2
import argparse
import time
from ultralytics import YOLO
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
YUNET    = os.path.expanduser("~/.cv2_models/face_detection_yunet_2023mar.onnx")

EMOTION_COLORS = {
    "Anger":     (0,   0,   220),
    "Contempt":  (0,   140, 140),
    "Disgust":   (0,   120, 120),
    "Fear":      (160, 0,   160),
    "Happiness": (0,   220, 0  ),
    "Neutral":   (180, 180, 180),
    "Sadness":   (200, 50,  50 ),
    "Surprise":  (0,   220, 220),
}

POSE_COLORS = {
    "Standing":    (0,   255, 128),
    "Sitting":     (255, 200, 0  ),
    "Arms Raised": (0,   180, 255),
    "Bending":     (255, 100, 0  ),
    "Leaning":     (200, 0,   200),
}

# COCO 17-keypoint indices
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
# 5:l_sh 6:r_sh 7:l_el 8:r_el 9:l_wr 10:r_wr
# 11:l_hip 12:r_hip 13:l_kn 14:r_kn 15:l_ank 16:r_ank

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

EMOTION_SKIP = 3
KPT_THRESH   = 0.3


# ── face helpers ──────────────────────────────────────────────────────────────
def make_face_detector(input_size=(320, 320)):
    return cv2.FaceDetectorYN.create(
        YUNET, "", input_size, score_threshold=0.5, nms_threshold=0.3)


def get_face_crop(face_det, crop):
    h, w = crop.shape[:2]
    face_det.setInputSize((w, h))
    _, faces = face_det.detect(crop)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[-1])
    x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    pad = 6
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(w, x+fw+pad), min(h, y+fh+pad)
    if x2-x1 < 20 or y2-y1 < 20:
        return None
    face = crop[y1:y2, x1:x2]
    fh2, fw2 = face.shape[:2]
    if fh2 < 96 or fw2 < 96:
        scale = max(96/fh2, 96/fw2)
        face = cv2.resize(face, (int(fw2*scale), int(fh2*scale)),
                          interpolation=cv2.INTER_LINEAR)
    return face


# ── keypoint helpers ──────────────────────────────────────────────────────────
def _pt(kpts, idx):
    """Return (x, y) if keypoint visible, else None."""
    if kpts is None or idx >= len(kpts):
        return None
    x, y, c = kpts[idx]
    return (float(x), float(y)) if float(c) > KPT_THRESH else None

def _conf(kpts, idx):
    """Return raw confidence for a keypoint."""
    return float(kpts[idx][2]) if kpts is not None and idx < len(kpts) else 0.0


# ── pose classifier ───────────────────────────────────────────────────────────
def classify_pose(kpts):
    """Overall body pose: Standing / Sitting / Arms Raised / Bending / Leaning."""
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
    """
    Returns head direction:
      Fwd | Head Down | Turned Left | Turned Right
    Uses nose vs eye positions for down, ear visibility for turns.
    """
    nose  = _pt(kpts, 0)
    l_eye = _pt(kpts, 1)
    r_eye = _pt(kpts, 2)
    l_ear_c = _conf(kpts, 3)
    r_ear_c = _conf(kpts, 4)

    # Head down: nose drops significantly below eye level
    if nose and l_eye and r_eye:
        eye_y  = (l_eye[1] + r_eye[1]) / 2
        eye_dx = abs(l_eye[0] - r_eye[0])
        if nose[1] > eye_y + max(eye_dx * 0.55, 12):
            return "Head Down"

    # Turned: one ear becomes hidden
    # When head turns LEFT  → person's right ear hides (r_ear_c drops)
    # When head turns RIGHT → person's left ear hides  (l_ear_c drops)
    if r_ear_c < 0.2 and l_ear_c > 0.3:
        return "Turned Left"
    if l_ear_c < 0.2 and r_ear_c > 0.3:
        return "Turned Right"

    return "Fwd"


# ── hand position ─────────────────────────────────────────────────────────────
def detect_hand_positions(kpts):
    """
    Returns (left_hand_label, right_hand_label).
    Labels: Raised | Pocket | Near Feet | Behind | Crossed | Down | Mid
    """
    l_sh,  r_sh  = _pt(kpts, 5),  _pt(kpts, 6)
    l_wr,  r_wr  = _pt(kpts, 9),  _pt(kpts, 10)
    l_hip, r_hip = _pt(kpts, 11), _pt(kpts, 12)
    l_ank, r_ank = _pt(kpts, 15), _pt(kpts, 16)
    l_wr_c = _conf(kpts, 9)
    r_wr_c = _conf(kpts, 10)

    body_cx = (l_hip[0] + r_hip[0]) / 2 if l_hip and r_hip else None

    def classify(wrist, wrist_conf, shoulder, hip, ankle, is_left):
        # Wrist not visible — likely behind body
        if wrist_conf < 0.15:
            return "Behind"
        if wrist is None:
            return "Hidden"
        wx, wy = wrist

        # Raised above shoulder
        if shoulder and wy < shoulder[1] - 15:
            return "Raised"

        # Near feet / ankles
        if ankle and wy > ankle[1] - 70:
            return "Near Feet"

        # In pocket: wrist near hip zone
        if hip and abs(wy - hip[1]) < 50 and abs(wx - hip[0]) < 65:
            return "Pocket"

        # Crossed arms: wrist on opposite side of body centre
        if body_cx is not None:
            if is_left  and wx > body_cx + 30:
                return "Crossed"
            if not is_left and wx < body_cx - 30:
                return "Crossed"

        # Hanging down below hip
        if hip and wy > hip[1] + 25:
            return "Down"

        return "Mid"

    lp = classify(l_wr, l_wr_c, l_sh, l_hip, l_ank, True)
    rp = classify(r_wr, r_wr_c, r_sh, r_hip, r_ank, False)
    return lp, rp


# ── leg stance ────────────────────────────────────────────────────────────────
def detect_leg_stance(kpts):
    """
    Returns leg stance label:
      Normal | Wide Stance | Feet Together | L-Leg Up | R-Leg Up
    """
    l_hip, r_hip = _pt(kpts, 11), _pt(kpts, 12)
    l_ank, r_ank = _pt(kpts, 15), _pt(kpts, 16)

    if l_ank and r_ank:
        # One leg significantly raised
        dy = abs(l_ank[1] - r_ank[1])
        if dy > 80:
            return "L-Leg Up" if l_ank[1] < r_ank[1] else "R-Leg Up"

        # Stance width relative to hip width
        ankle_dx = abs(l_ank[0] - r_ank[0])
        if l_hip and r_hip:
            hip_w = max(abs(l_hip[0] - r_hip[0]), 1)
            ratio = ankle_dx / hip_w
            if ratio > 1.8:
                return "Wide Stance"
            if ratio < 0.3:
                return "Feet Together"

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


def _tag(frame, text, x, y, bg, text_color=(0, 0, 0), font_scale=0.45, thickness=1):
    """Draw a filled text tag, return bottom-y of tag."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    by = y + th + 5
    cv2.rectangle(frame, (x, y), (x + tw + 6, by), bg, -1)
    cv2.putText(frame, text, (x + 3, by - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return by


def draw_result(frame, box, idx, emotion, score, pose, head, l_hand, r_hand, legs):
    x1, y1, x2, y2 = box
    emo_c  = EMOTION_COLORS.get(emotion, (180,180,180)) if emotion else (80,80,80)
    pose_c = POSE_COLORS.get(pose, (160,160,160))

    # Bounding box
    cv2.rectangle(frame, (x1,y1), (x2,y2), emo_c, 2)

    # Emotion label — above box
    emo_lbl = f"#{idx} {emotion} {score:.0f}%" if emotion else f"#{idx} no face"
    (tw, th), _ = cv2.getTextSize(emo_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    ly = max(y1, th + 6)
    cv2.rectangle(frame, (x1, ly-th-6), (x1+tw+6, ly), emo_c, -1)
    cv2.putText(frame, emo_lbl, (x1+3, ly-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2)

    # Labels stacked below box
    cy = y2 + 2
    cy = _tag(frame, pose,                          x1, cy, pose_c)
    cy = _tag(frame, f"Head : {head}",              x1, cy, (30, 30, 50), (220, 220, 220))
    cy = _tag(frame, f"LH:{l_hand}  RH:{r_hand}",  x1, cy, (30, 30, 50), (220, 220, 220))
    cy = _tag(frame, f"Legs : {legs}",              x1, cy, (30, 30, 50), (220, 220, 220))


# ── source open ───────────────────────────────────────────────────────────────
def open_source(source):
    if source == "webcam":
        return cv2.VideoCapture(0), "Webcam"
    if source.startswith("http://") or source.startswith("rtsp://"):
        url = source.rstrip("/") + "/video" \
              if "8080" in source and not source.endswith("/video") else source
        print(f"[INFO] IP camera: {url}")
        return cv2.VideoCapture(url), "IP Cam"
    if not os.path.exists(source):
        raise FileNotFoundError(f"Not found: {source}")
    return cv2.VideoCapture(source), os.path.basename(source)


# ── main loop ─────────────────────────────────────────────────────────────────
def run(source, save_path=None, show=True):
    print(f"[INFO] Device   : {DEVICE}  ({GPU_NAME})")
    print(f"[INFO] Pipeline : YOLOv8n-Pose → YuNet → HSEmotion(ONNX)")
    print(f"[INFO] Analysis : Emotion + Pose + Head + Hands + Legs")

    yolo          = YOLO("yolov8n-pose.pt")
    emotion_model = HSEmotionRecognizer(model_name="enet_b2_8")
    face_det      = make_face_detector()

    cap, src_label = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source   : {src_label} | {W}x{H} @ {fps_in:.0f}fps | q = quit")

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
        print(f"[INFO] Saving   : {save_path}")

    frame_idx  = 0
    # cached: i -> (emotion, score, pose, head, l_hand, r_hand, legs, kpts)
    cached     = {}
    prev_boxes = []
    prev_kpts  = []
    fps_t, fps_n, disp_fps = time.time(), 0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLOv8-Pose: person + keypoints every frame ───────────────────
        out = yolo(frame, classes=[0], verbose=False, device=DEVICE)
        boxes, kpts_list = [], []
        if out and out[0].boxes is not None:
            for i, b in enumerate(out[0].boxes):
                if float(b.conf[0]) >= 0.35:
                    boxes.append(tuple(map(int, b.xyxy[0])))
                    if out[0].keypoints is not None and i < len(out[0].keypoints.data):
                        kpts_list.append(out[0].keypoints.data[i].cpu().numpy())
                    else:
                        kpts_list.append(None)
        prev_boxes = boxes
        prev_kpts  = kpts_list

        # ── Full analysis every EMOTION_SKIP frames ───────────────────────
        if frame_idx % EMOTION_SKIP == 0:
            new_cache = {}
            for i, (x1,y1,x2,y2) in enumerate(boxes):
                kpts = prev_kpts[i] if i < len(prev_kpts) else None

                pose   = classify_pose(kpts)
                head   = detect_head_direction(kpts)
                l_hand, r_hand = detect_hand_positions(kpts)
                legs   = detect_leg_stance(kpts)

                person_crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                if person_crop.size == 0:
                    new_cache[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)
                    continue

                face = get_face_crop(face_det, person_crop)
                if face is None:
                    new_cache[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)
                    continue

                try:
                    emotion, scores = emotion_model.predict_emotions(face, logits=False)
                    new_cache[i] = (emotion, float(max(scores))*100,
                                    pose, head, l_hand, r_hand, legs, kpts)
                except Exception:
                    new_cache[i] = (None, 0, pose, head, l_hand, r_hand, legs, kpts)

            cached = new_cache

        # ── Draw ──────────────────────────────────────────────────────────
        annotated = frame.copy()
        for i, box in enumerate(prev_boxes):
            emo, sc, pose, head, l_hand, r_hand, legs, kpts = cached.get(
                i, (None, 0, "Standing", "Fwd", "Mid", "Mid", "Normal", None))
            draw_skeleton(annotated, kpts)
            draw_result(annotated, box, i+1, emo, sc, pose, head, l_hand, r_hand, legs)

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            disp_fps    = fps_n / (time.time() - fps_t)
            fps_n, fps_t = 0, time.time()

        cv2.putText(annotated, f"FPS:{disp_fps:.1f}  People:{len(prev_boxes)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Behaviour + Pose Analyser", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  required=True,
                    help="video path / 'webcam' / 'http://ip:8080' / 'rtsp://...'")
    ap.add_argument("--save",    default=None,  help="output video path")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    run(source=args.source, save_path=args.save, show=not args.no_show)
