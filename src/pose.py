import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# Mapeamento dos 33 landmarks do MediaPipe Pose
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# Conexões padrão do corpo
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
}

def init_pose_detector(model_type="heavy"):
    """
    Garante o download do modelo selecionado (lite, full, heavy)
    e inicializa o PoseLandmarker com rastreamento temporal (VIDEO mode).
    """
    model_type = model_type.lower()
    if model_type not in MODEL_URLS:
        raise ValueError(f"Modelo inválido: '{model_type}'. Escolha entre 'lite', 'full' ou 'heavy'.")

    model_path = f"models/pose_landmarker_{model_type}.task"

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = MODEL_URLS[model_type]
        print(f"Baixando modelo '{model_type}' a partir de {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download concluído.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return PoseLandmarker.create_from_options(options)

def detect_pose(detector, frame_rgb, timestamp_ms):
    """Recebe o frame RGB e seu timestamp em ms para manter a coerência temporal."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        return result.pose_landmarks[0]
    return None