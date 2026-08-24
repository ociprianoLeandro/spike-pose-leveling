"""
pose.py

Responsabilidade única: rodar a detecção de pose sobre um frame e
devolver os landmarks detectados (ou None, se ninguém for detectado).

Este módulo não sabe nada sobre vídeo ou desenho — apenas detecção.

Nota sobre a API utilizada:
O MediaPipe migrou da antiga API "mp.solutions.pose" para a "Tasks API"
(mp.tasks.python.vision.PoseLandmarker), que é a forma atual e suportada
de usar Pose Estimation. A diferença prática é que essa API precisa de
um arquivo de modelo (.task) carregado do disco -- por isso baixamos
esse arquivo automaticamente na primeira execução.
"""

import os
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Modelo "lite": bom equilíbrio entre velocidade e precisão para começar.
# Outras opções oficiais: pose_landmarker_full.task e pose_landmarker_heavy.task
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")


def ensure_model_downloaded(model_path=MODEL_PATH, url=MODEL_URL):
    """Garante que o arquivo de modelo exista localmente, baixando-o se necessário.

    O modelo não é versionado no Git (é um binário de ~5-30MB) -- ele é
    baixado uma vez e reaproveitado nas próximas execuções.
    """
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Modelo de pose não encontrado. Baixando de:\n  {url}")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível baixar o modelo de pose automaticamente "
            f"({exc}).\n"
            "Baixe manualmente o arquivo em:\n"
            f"  {url}\n"
            f"e salve-o em: {model_path}"
        ) from exc
    print(f"Modelo salvo em: {model_path}")
    return model_path


class Landmark:
    """Estrutura simples para representar um landmark, mantendo a mesma
    interface (x, y, z, visibility) usada no restante do projeto."""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility if visibility is not None else 1.0


class PoseLandmarks:
    """Agrupa a lista de landmarks de uma pessoa detectada, com a mesma
    interface `.landmark` usada pela API antiga do MediaPipe."""

    def __init__(self, raw_landmarks):
        self.landmark = [
            Landmark(lm.x, lm.y, lm.z, lm.visibility) for lm in raw_landmarks
        ]


class PoseDetector:
    def __init__(self, model_path=None, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        model_path = model_path or ensure_model_downloaded()

        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

        # Conexões (ossos) entre os 33 landmarks do corpo
        self._connections = [
            (c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
        ]
        self._landmark_names = vision.PoseLandmark

    def process(self, frame_rgb):
        """Recebe um frame em RGB (numpy array) e retorna os landmarks
        da primeira pessoa detectada, ou None se ninguém for detectado.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        # A API de VIDEO exige timestamps crescentes em milissegundos
        self._timestamp_ms += 33  # aproximação (~30fps); não precisa ser exato
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            return None

        # Pegamos a primeira pessoa detectada (o modelo suporta múltiplas)
        return PoseLandmarks(result.pose_landmarks[0])

    def close(self):
        self.landmarker.close()

    @property
    def connections(self):
        """Pares de índices (start, end) que devem ser ligados por uma linha."""
        return self._connections

    @property
    def landmark_names(self):
        """Enum com o nome de cada landmark (ex: LEFT_SHOULDER, LEFT_ELBOW...)."""
        return self._landmark_names