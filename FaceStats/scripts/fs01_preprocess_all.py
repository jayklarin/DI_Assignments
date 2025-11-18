import os
import cv2
import json
import time
import glob
import torch
import traceback
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image
import mediapipe as mp

# ============================================================
# PATHS — You can edit RAW_DIR only
# ============================================================

RAW_DIR = "/Users/jayklarin/__DI/Repositories/DI_Assignments/FaceStats/dataset_raw/part4_sd21/images"
OUT_DIR = "/Users/jayklarin/__DI/Repositories/DI_Assignments/FaceStats/dataset_preprocessed"
META_DIR = "/Users/jayklarin/__DI/Repositories/DI_Assignments/FaceStats/dataset_metadata"
LOG_DIR = "/Users/jayklarin/__DI/Repositories/DI_Assignments/FaceStats/logs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "preprocess.log")
fail_file = os.path.join(LOG_DIR, "failures.log")

# ============================================================
# MEDIAPIPE MODELS
# ============================================================

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=3,
    min_detection_confidence=0.5
)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=4,
    min_detection_confidence=0.4
)

USE_MPS = torch.backends.mps.is_available()

# ============================================================
# LOGGING
# ============================================================

def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")

def log_fail(msg):
    with open(fail_file, "a") as f:
        f.write(msg + "\n")

def to_mps(t):
    return t.to("mps") if USE_MPS else t

# ============================================================
# FAST PATH HEURISTICS
# ============================================================

def fast_path_ok(img):
    """
    Quick symmetry/brightness check to skip heavy Mediapipe path.
    """
    h, w = img.shape[:2]

    left = img[:, :w//2].mean()
    right = img[:, w//2:].mean()
    if abs(left - right) > 12:
        return False

    if img.std() < 12:
        return False

    return True

# ============================================================
# SQUARE-CROP (FIXED)
# ============================================================

def square_crop(img):
    h_c, w_c = img.shape[:2]
    side = max(h_c, w_c)

    pad_top = (side - h_c) // 2
    pad_bottom = side - h_c - pad_top
    pad_left = (side - w_c) // 2
    pad_right = side - w_c - pad_left

    return cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_REFLECT_101
    )

# ============================================================
# FULL PROCESS FUNCTION
# ============================================================

def process_one(path):
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f"{base}.png")
        meta_path = os.path.join(META_DIR, f"{base}.json")

        # Skip if exists
        if os.path.exists(out_path):
            return "skip"

        img = cv2.imread(path)
        if img is None:
            log_fail(f"{base}: could not read file")
            return "fail"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --------------------------------------------------------
        # FAST PATH
        # --------------------------------------------------------
        if fast_path_ok(img_rgb):
            crop = square_crop(img)
            final = cv2.resize(crop, (512, 512), cv2.INTER_LANCZOS4)
            Image.fromarray(final).save(out_path)

            meta = {
                "id": base,
                "raw": path,
                "method": "fast_path",
                "resolution": "512x512"
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            return "fast"

        # --------------------------------------------------------
        # FULL MEDIAPIPE PATH
        # --------------------------------------------------------
        face_res = mp_face.process(img_rgb)
        hand_res = mp_hands.process(img_rgb)

        if not face_res.multi_face_landmarks:
            log_fail(f"{base}: no face detected")
            return "fail"

        if len(face_res.multi_face_landmarks) > 1:
            log_fail(f"{base}: multiple faces detected")
            return "fail"

        lms = face_res.multi_face_landmarks[0].landmark

        # Eye alignment
        left_eye = np.mean([[lms[i].x * w, lms[i].y * h] for i in [33, 133]], axis=0)
        right_eye = np.mean([[lms[i].x * w, lms[i].y * h] for i in [362, 263]], axis=0)

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (w, h))

        rot_lm = []
        for lm in lms:
            px = M[0,0]*(lm.x*w) + M[0,1]*(lm.y*h) + M[0,2]
            py = M[1,0]*(lm.x*w) + M[1,1]*(lm.y*h) + M[1,2]
            rot_lm.append([px, py])
        rot_lm = np.array(rot_lm)

        xs = rot_lm[:, 0]
        ys = rot_lm[:, 1]

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.35)

        cx1 = max(x1 - pad_x, 0)
        cx2 = min(x2 + pad_x, w)
        cy1 = max(y1 - pad_y, 0)
        cy2 = min(y2 + pad_y, h)

        crop = aligned[cy1:cy2, cx1:cx2]
        crop_sq = square_crop(crop)

        final = cv2.resize(crop_sq, (512, 512), cv2.INTER_LANCZOS4)
        Image.fromarray(final).save(out_path)

        meta = {
            "id": base,
            "raw": path,
            "method": "full_path",
            "angle": angle,
            "crop_box": [cx1, cy1, cx2, cy2],
            "resolution": "512x512"
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return "full"

    except Exception as e:
        log_fail(f"ERROR processing {path}: {str(e)}")
        traceback.print_exc()
        return "fail"

# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    # RECURSIVE SCAN (fix for nested folders)
    files = glob.glob(f"{RAW_DIR}/**/*.jpg", recursive=True)
    files += glob.glob(f"{RAW_DIR}/**/*.png", recursive=True)

    print(f"Found {len(files)} images in RAW_DIR")
    log(f"Starting preprocessing: {len(files)} images")

    # CPU OR SINGLE THREAD
    for path in tqdm(files, desc="Processing"):
        process_one(path)

    print("Done.")
    log("Completed preprocessing run.")
