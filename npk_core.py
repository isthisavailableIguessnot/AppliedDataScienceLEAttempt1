"""Core logic for the LeafNPK app: model loading, prediction, Grad-CAM, photo checks.

Kept separate from the Streamlit UI so it can be tested and benchmarked on its own.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "efficientnetb0.keras"
CARD_PATH = APP_DIR / "model" / "efficientnetb0_card.json"
FERT_PATH = APP_DIR / "fertilizer_info.json"

# ---------------------------------------------------------------------------
# Photo-quality thresholds.
# The ML robustness tests showed the pretrained models hold up under moderate
# lighting changes but fail on grayscale input and strong overexposure (x1.30).
# These checks warn the user before they trust a result from such a photo.
# ---------------------------------------------------------------------------
CLIPPED_PIXEL_MAX = 0.08     # share of pixels with any channel >= 250 (blown highlights)
BRIGHTNESS_MAX = 185.0       # mean luminance (0-255) above this = very bright / washed out
BRIGHTNESS_MIN = 45.0        # mean luminance below this = too dark
SATURATION_MIN = 0.08        # mean HSV saturation below this = (near) grayscale
MIN_SIDE_PX = 160            # images smaller than this lose leaf detail


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model():
    """Load the Keras model and split it into backbone + classification head.

    The saved model is: image -> backbone (EfficientNet-B0, includes its own
    preprocessing) -> gap -> dropout -> predictions. Running the backbone and
    head separately lets one forward pass give both the prediction and the
    Grad-CAM heatmap.
    """
    import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)

    # Streamlit calls keras.backend.clear_session() after every script run
    # (streamlit/runtime/scriptrunner/script_runner.py: _clean_problem_modules).
    # That clears TensorFlow's kernel cache, so the next prediction rebuilds its
    # kernels and takes ~0.7 s instead of ~30 ms. This app loads one model once
    # and never builds new ones, so the cleanup is not needed: make it a no-op.
    keras.backend.clear_session = lambda *args, **kwargs: None
    backbone = model.get_layer("backbone")
    head = [model.get_layer(n) for n in ("gap", "dropout", "predictions")]
    return model, backbone, head


def prepare_image(file_or_img, img_size: int = 224) -> tuple[Image.Image, np.ndarray]:
    """Open an upload, fix phone rotation, convert to RGB, resize like training.

    Returns the display image (original size, upright RGB) and a float32 batch
    of shape (1, img_size, img_size, 3) with raw 0-255 values.
    """
    img = file_or_img if isinstance(file_or_img, Image.Image) else Image.open(file_or_img)
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = np.asarray(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)
    return img, x[None, ...]


@dataclass
class QualityReport:
    brightness: float
    clipped_share: float
    saturation: float
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)


def check_quality(img: Image.Image) -> QualityReport:
    small = img.copy()
    small.thumbnail((512, 512))
    rgb = np.asarray(small, dtype=np.float32)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    clipped = float((rgb.max(axis=-1) >= 250).mean())
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    sat = float(hsv[..., 1].mean() / 255.0)
    w, h = img.size

    r = QualityReport(float(lum.mean()), clipped, sat, w, h)
    if clipped > CLIPPED_PIXEL_MAX or r.brightness > BRIGHTNESS_MAX:
        r.warnings.append(
            "The photo looks overexposed (too bright or washed out). In testing, the model "
            "made many more mistakes on overexposed leaves. Retake it in shade or away from "
            "direct sun."
        )
    if r.brightness < BRIGHTNESS_MIN:
        r.warnings.append("The photo is very dark. Retake it in better light.")
    if sat < SATURATION_MIN:
        r.warnings.append(
            "The photo has almost no colour (black-and-white or heavily filtered). The model "
            "relies on leaf colour and performed poorly on grayscale images. Use a normal "
            "colour photo."
        )
    if min(w, h) < MIN_SIDE_PX:
        r.warnings.append(
            f"The image is small ({w}x{h} px). Use a closer, sharper photo of the leaf."
        )
    return r


def build_predictor(backbone, head, img_size: int = 224):
    """Return a compiled function x -> (probs, cam) for the top class.

    One forward pass gives both the prediction and the Grad-CAM map. Compiling
    with tf.function makes it about 20x faster than eager mode on CPU; the first
    call takes a few seconds to trace, so the app warms it up at start-up.
    """
    import tensorflow as tf

    gap, dropout, dense = head

    @tf.function(input_signature=[tf.TensorSpec([1, img_size, img_size, 3], tf.float32)])
    def run(x):
        with tf.GradientTape() as tape:
            feats = backbone(x, training=False)
            tape.watch(feats)
            probs = dense(dropout(gap(feats), training=False))
            score = tf.gather(probs[0], tf.argmax(probs[0]))
        grads = tape.gradient(score, feats)
        weights = tf.reduce_mean(grads, axis=(1, 2))                 # (1, C)
        cam = tf.nn.relu(tf.reduce_sum(feats * weights[:, None, None, :], axis=-1))[0]
        return probs[0], cam

    run(tf.zeros([1, img_size, img_size, 3]))  # warm-up / trace once
    return run


def predict_with_gradcam(predictor, x: np.ndarray):
    """-> (probabilities, Grad-CAM heatmap 7x7 normalised to 0-1, milliseconds)."""
    t0 = time.perf_counter()
    probs, cam = predictor(x.astype(np.float32))
    probs, cam = probs.numpy(), cam.numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    ms = (time.perf_counter() - t0) * 1000.0
    return probs, cam, ms


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize the 7x7 CAM to the photo's size and blend it with a jet colormap."""
    from matplotlib import colormaps

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(img, dtype=np.float32) / 255.0
    # Fade the colour out where the model paid little attention, so the leaf stays visible.
    a = alpha * np.clip(np.asarray(cam_img, dtype=np.float32) / 255.0, 0.0, 1.0)[..., None] ** 0.7
    out = (1 - a) * base + a * heat
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


def cost_per_kg_nutrient(price_per_bag: float | None, pct: float, bag_kg: float = 50.0):
    """PHP per kg of the nutrient (N, P2O5 or K2O) delivered by one product."""
    if price_per_bag is None or pct <= 0:
        return None
    return price_per_bag / (bag_kg * pct / 100.0)
