"""
Konversi best_gesture_model.keras → best_gesture_model.onnx
Menggunakan tf2onnx.convert.from_function (Keras 3.x compatible)

Jalankan SEKALI SAJA:
    .venv\Scripts\python.exe convert_to_onnx.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

BASE_DIR        = Path(__file__).resolve().parent
KERAS_PATH      = BASE_DIR / "models" / "best_gesture_model.keras"
ONNX_PATH       = BASE_DIR / "models" / "best_gesture_model.onnx"

NUM_FRAMES         = 30
FEATURES_PER_FRAME = 225

# ── Step 1: Muat model ────────────────────────────────────────────
print(f"[1/3] Memuat model: {KERAS_PATH}")
model = tf.keras.models.load_model(str(KERAS_PATH))
print(f"      Input  : {model.input_shape}")
print(f"      Output : {model.output_shape}")

# ── Step 2: Buat concrete function lalu konversi ──────────────────
print(f"\n[2/3] Membuat TF concrete function...")

INPUT_SPEC = tf.TensorSpec(
    shape=(None, NUM_FRAMES, FEATURES_PER_FRAME),
    dtype=tf.float32,
    name="input_1"
)

@tf.function(input_signature=[INPUT_SPEC])
def serving_fn(x):
    return model(x, training=False)

# Materialize concrete function
cf = serving_fn.get_concrete_function(INPUT_SPEC)
print(f"      Inputs : {[i.name for i in cf.inputs]}")
print(f"      Outputs: {[o.name for o in cf.outputs]}")

# ── Step 3: Konversi ke ONNX ─────────────────────────────────────
print(f"\n[3/3] Konversi ke ONNX: {ONNX_PATH}")

onnx_model, _ = tf2onnx.convert.from_function(
    serving_fn,
    input_signature=[INPUT_SPEC],
    opset=17,
    output_path=str(ONNX_PATH),
)

print(f"\n[OK] Model ONNX tersimpan: {ONNX_PATH}")
print(f"     Ukuran: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# ── Verifikasi ────────────────────────────────────────────────────
print("\n[VER] Verifikasi ONNX model (CPU)...")
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
inp_meta  = sess.get_inputs()[0]
out_meta  = sess.get_outputs()[0]
print(f"     Input : '{inp_meta.name}' shape={inp_meta.shape}")
print(f"     Output: '{out_meta.name}' shape={out_meta.shape}")

dummy  = np.zeros((1, NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)
result = sess.run([out_meta.name], {inp_meta.name: dummy})[0]
print(f"     Test inference OK — output: {result.shape}")

print()
print("=" * 55)
print("Konversi berhasil! Sekarang jalankan GPU inference:")
print("  .venv\\Scripts\\python.exe realtime_inference_onnx.py")
print("=" * 55)
