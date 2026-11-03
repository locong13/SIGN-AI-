
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Paksa stdout UTF-8 agar karakter Unicode tampil di terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import json
import time
from pathlib import Path
from collections import deque

# ─────────────────────────────────────────────
# 0. KONFIGURASI ONNX RUNTIME + GPU
# ─────────────────────────────────────────────
def setup_ort_session(model_path: str):
    """
    Buat ONNX Runtime session dengan provider terbaik:
    1. DmlExecutionProvider   → GPU (DirectML / RTX 4050)
    2. CPUExecutionProvider   → fallback CPU
    """
    available = ort.get_available_providers()
    print(f"[ORT] Provider tersedia: {available}")

    # Coba DirectML (GPU Windows)
    if "DmlExecutionProvider" in available:
        providers = [
            ("DmlExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        session = ort.InferenceSession(model_path, providers=providers)
        active  = session.get_providers()
        device  = "GPU:RTX4050 (DirectML)" if "DmlExecutionProvider" in active else "CPU"
    else:
        providers = ["CPUExecutionProvider"]
        session   = ort.InferenceSession(model_path, providers=providers)
        device    = "CPU"

    print(f"[ORT] Device aktif : {device}")
    print(f"[ORT] Provider aktif: {session.get_providers()}")
    return session, device


# ─────────────────────────────────────────────
# 1. KONFIGURASI PATH
# ─────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
ONNX_PATH    = BASE_DIR / "models" / "best_gesture_model.onnx"
CONFIG_PATH  = BASE_DIR / "models" / "config.json"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"Model ONNX tidak ditemukan: {ONNX_PATH}\n"
        "Jalankan dulu: .venv\\Scripts\\python.exe convert_to_onnx.py"
    )

# ─────────────────────────────────────────────
# 2. MUAT CONFIG
# ─────────────────────────────────────────────
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

NUM_FRAMES         = cfg["num_frames"]           # 30
FEATURES_PER_FRAME = cfg["features_per_frame"]   # 225
MIN_DET_CONF       = cfg["min_detection_confidence"]
MIN_TRACK_CONF     = cfg["min_tracking_confidence"]
LABEL_TO_GLOSS     = {int(k): v for k, v in cfg["label_to_gloss"].items()}
NUM_CLASSES        = cfg["num_classes"]           # 32

print(f"[INFO] Config: {NUM_FRAMES} frame x {FEATURES_PER_FRAME} fitur -> {NUM_CLASSES} kelas")

# ─────────────────────────────────────────────
# 3. MUAT ORT SESSION + LABEL ENCODER
# ─────────────────────────────────────────────
print(f"[INFO] Memuat model ONNX: {ONNX_PATH}")
ort_session, DEVICE_LABEL = setup_ort_session(str(ONNX_PATH))

# Nama input/output dari ONNX graph
INPUT_NAME  = ort_session.get_inputs()[0].name
OUTPUT_NAME = ort_session.get_outputs()[0].name
print(f"[ORT] Input  : '{INPUT_NAME}'  shape={ort_session.get_inputs()[0].shape}")
print(f"[ORT] Output : '{OUTPUT_NAME}' shape={ort_session.get_outputs()[0].shape}")

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
print(f"[INFO] Label encoder dimuat — {len(label_encoder.classes_)} kelas")

# ─────────────────────────────────────────────
# 4. WARMUP SESSION
# ─────────────────────────────────────────────
_dummy = np.zeros((1, NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)
_ = ort_session.run([OUTPUT_NAME], {INPUT_NAME: _dummy})
print("[INFO] Warmup selesai — inferensi GPU siap\n")

# ─────────────────────────────────────────────
# 5. INISIALISASI MEDIAPIPE
# ─────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────
# 6. FUNGSI HELPER
# ─────────────────────────────────────────────
def landmarks_array(landmarks, n_points: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((n_points, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32)


def frame_to_features(result) -> np.ndarray:
    pose  = landmarks_array(result.pose_landmarks,       33).flatten()
    left  = landmarks_array(result.left_hand_landmarks,  21).flatten()
    right = landmarks_array(result.right_hand_landmarks, 21).flatten()
    return np.concatenate([pose, left, right])  # (225,)


def predict_gesture(sequence: np.ndarray):
    """sequence: (30, 225) → (gloss, confidence)"""
    x     = sequence[np.newaxis, ...].astype(np.float32)   # (1, 30, 225)
    probs = ort_session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]  # (32,)
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label_id   = int(label_encoder.inverse_transform([class_idx])[0])
    gloss      = LABEL_TO_GLOSS.get(label_id, f"Label-{label_id}")
    return gloss, confidence


# ─────────────────────────────────────────────
# 7. PARAMETER TAMPILAN
# ─────────────────────────────────────────────
FONT      = cv2.FONT_HERSHEY_DUPLEX
GREEN     = (0, 230, 0)
RED       = (0, 0, 220)
YELLOW    = (0, 220, 220)
CYAN      = (220, 220, 0)
WHITE     = (255, 255, 255)
DARK      = (30, 30, 30)
BAR_COLOR = (50, 180, 255)
GPU_COLOR = (0, 165, 255)   # oranye terang = GPU aktif

CONF_THRESHOLD = 0.50

# ─────────────────────────────────────────────
# 8. AUTO-DETEKSI KAMERA (DirectShow)
# ─────────────────────────────────────────────
def find_camera(preferred_index: int = 0):
    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Kamera ditemukan di index {idx} (DirectShow)")
                return cap, idx
            cap.release()
    raise RuntimeError("Tidak ada kamera yang dapat dibuka.")


# ─────────────────────────────────────────────
# 9. LOOP KAMERA UTAMA
# ─────────────────────────────────────────────
def run_realtime(camera_index: int = 0) -> None:
    cap, actual_index = find_camera(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_buffer    = deque(maxlen=NUM_FRAMES)
    is_recording    = False
    prediction_text = "—"
    confidence_val  = 0.0
    status_msg      = "Tekan [SPACE] untuk mulai rekam"
    rec_start_time  = 0.0
    history: list   = []
    fail_count      = 0

    print("[INFO] Kamera dibuka. [SPACE]=Rekam  [R]=Reset  [Q]=Keluar\n")

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    ) as holistic:

        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                fail_count += 1
                if fail_count % 30 == 1:
                    print(f"[WARN] Frame tidak terbaca ({fail_count}x)…")
                if fail_count > 300:
                    cap.release(); time.sleep(1.0)
                    cap, _ = find_camera(actual_index)
                    fail_count = 0
                time.sleep(0.05)
                continue
            fail_count = 0

            bgr = cv2.flip(bgr, 1)
            h, w = bgr.shape[:2]

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)
            rgb.flags.writeable = True

            # Gambar landmark
            mp_drawing.draw_landmarks(bgr, result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(bgr, result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(bgr, result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 200, 0), thickness=1, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(150, 150, 0), thickness=1))

            # Rekam frame
            if is_recording:
                frame_buffer.append(frame_to_features(result))
                recorded  = len(frame_buffer)
                bar_width = int((recorded / NUM_FRAMES) * (w - 40))
                elapsed   = time.time() - rec_start_time
                cv2.rectangle(bgr, (20, h - 40), (w - 20, h - 20), DARK, -1)
                cv2.rectangle(bgr, (20, h - 40), (20 + bar_width, h - 20), BAR_COLOR, -1)
                cv2.putText(bgr, f"Merekam: {recorded}/{NUM_FRAMES}  ({elapsed:.1f}s)",
                            (25, h - 24), FONT, 0.5, WHITE, 1)

                if recorded >= NUM_FRAMES:
                    is_recording = False
                    seq = np.array(frame_buffer, dtype=np.float32)
                    pred_gloss, pred_conf = predict_gesture(seq)
                    if pred_conf >= CONF_THRESHOLD:
                        prediction_text = pred_gloss
                        confidence_val  = pred_conf
                        history.append(pred_gloss)
                        if len(history) > 5: history.pop(0)
                        status_msg = "Prediksi selesai! Tekan [SPACE] untuk rekam lagi"
                    else:
                        prediction_text = f"Tidak yakin ({pred_gloss})"
                        confidence_val  = pred_conf
                        status_msg = f"Confidence rendah ({pred_conf:.0%}) — coba lagi"
                    print(f"[PRED] {pred_gloss}  conf={pred_conf:.1%}")

            # ── UI Overlay ──────────────────────────────────────────
            overlay = bgr.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), DARK, -1)
            cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)

            cv2.putText(bgr, "SignBridge AI — BISINDO Realtime",
                        (16, 28), FONT, 0.75, CYAN, 1)
            cv2.putText(bgr, f"[{DEVICE_LABEL}]",
                        (w - 250, 28), FONT, 0.52, GPU_COLOR, 1)
            cv2.putText(bgr, time.strftime("%H:%M:%S"),
                        (w - 110, 52), FONT, 0.55, WHITE, 1)
            cv2.putText(bgr, status_msg, (16, 62), FONT, 0.52,
                        RED if is_recording else YELLOW, 1)

            # Panel prediksi
            px, py, pw, ph = 16, 95, 360, 95
            ov2 = bgr.copy()
            cv2.rectangle(ov2, (px, py), (px+pw, py+ph), DARK, -1)
            cv2.addWeighted(ov2, 0.6, bgr, 0.4, 0, bgr)
            cv2.rectangle(bgr, (px, py), (px+pw, py+ph), CYAN, 1)
            cv2.putText(bgr, "Prediksi:", (px+10, py+24), FONT, 0.52, CYAN, 1)
            cv2.putText(bgr, prediction_text, (px+10, py+58), FONT, 0.85, GREEN, 2)
            bar_max  = pw - 20
            bar_fill = int(confidence_val * bar_max)
            cv2.rectangle(bgr, (px+10, py+70), (px+10+bar_max, py+84), DARK, -1)
            bar_c = GREEN if confidence_val >= CONF_THRESHOLD else RED
            cv2.rectangle(bgr, (px+10, py+70), (px+10+bar_fill, py+84), bar_c, -1)
            cv2.putText(bgr, f"{confidence_val:.0%}",
                        (px+bar_max-30, py+83), FONT, 0.42, WHITE, 1)

            # Riwayat
            if history:
                hy = 200
                cv2.putText(bgr, "Riwayat:", (px+10, hy), FONT, 0.48, YELLOW, 1)
                for i, h_pred in enumerate(reversed(history)):
                    alpha = max(0.2, 1.0 - i * 0.18)
                    color = tuple(int(c * alpha) for c in YELLOW)
                    cv2.putText(bgr, f"  {h_pred}", (px+10, hy+22*(i+1)),
                                FONT, 0.50, color, 1)

            # Indikator tangan
            hl = result.left_hand_landmarks  is not None
            hr = result.right_hand_landmarks is not None
            cv2.putText(bgr, f"Kiri : {'OK' if hl else '---'}",
                        (w-180, 95+22), FONT, 0.50, GREEN if hl else RED, 1)
            cv2.putText(bgr, f"Kanan: {'OK' if hr else '---'}",
                        (w-180, 95+46), FONT, 0.50, GREEN if hr else RED, 1)

            # Hint
            hint_y = h - 60 if not is_recording else h - 70
            cv2.putText(bgr, "[SPACE] Rekam  [R] Reset  [Q] Keluar",
                        (16, hint_y), FONT, 0.45, WHITE, 1)

            cv2.imshow("SignBridge AI — BISINDO GPU (ONNX)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                frame_buffer.clear()
                is_recording   = True
                rec_start_time = time.time()
                status_msg     = "Merekam…"
                print("[INFO] Rekaman dimulai…")
            elif key in (ord('r'), ord('R')):
                frame_buffer.clear()
                is_recording    = False
                prediction_text = "—"
                confidence_val  = 0.0
                status_msg      = "Buffer direset. Tekan [SPACE] untuk rekam"

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


# ─────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge AI — GPU ONNX Inference")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index kamera (default: 0)")
    args = parser.parse_args()
    run_realtime(camera_index=args.camera)
