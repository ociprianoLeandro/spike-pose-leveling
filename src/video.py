"""
video.py

Responsabilidade única: abrir uma fonte de vídeo (arquivo ou webcam),
ler frames e escrever o vídeo processado em disco.

Este módulo não sabe nada sobre MediaPipe ou desenho de skeleton —
ele só lida com entrada/saída de vídeo via OpenCV.
"""

import cv2


class VideoSource:
    """Wrapper em torno do cv2.VideoCapture.

    `source` pode ser:
    - um caminho de arquivo (string), ex: "input/ataque.mp4"
    - um índice de webcam (int), normalmente 0
    """

    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo: {source}"
            )

    @property
    def width(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        # Algumas webcams retornam 0 — usamos um valor padrão seguro
        return fps if fps and fps > 0 else 30.0

    @property
    def frame_count(self):
        # Para webcam isso normalmente retorna 0 ou um valor não confiável
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self):
        """Lê o próximo frame. Retorna None quando o vídeo termina
        ou a leitura falha (ex: webcam desconectada)."""
        success, frame = self.capture.read()
        if not success:
            return None
        return frame

    def release(self):
        self.capture.release()


class VideoWriter:
    """Wrapper em torno do cv2.VideoWriter para salvar o vídeo processado."""

    def __init__(self, output_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Não foi possível criar o arquivo de saída: {output_path}")

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()