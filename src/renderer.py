import cv2
import numpy as np

def draw_skeleton_on_blank(landmarks, connections, width, height, min_visibility=0.5):
    """Cria uma tela preta e desenha os pontos e linhas identificados."""

    # primeiro cria a tela preta com a mesma resolução do vídeo original
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if landmarks is None:
        return canvas

    # Depois se converte as coordenadas normalizadas (0 a 1) para pixels
    points = {}
    for idx, lm in enumerate(landmarks):
        if lm.visibility >= min_visibility:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points[idx] = (x, y)

    # Desenha as arestad
    for start, end in connections:
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], (0, 255, 0), 2)

    # Desenha os vértices/articulações
    for pt in points.values():
        cv2.circle(canvas, pt, 4, (0, 0, 255), -1)

    return canvas