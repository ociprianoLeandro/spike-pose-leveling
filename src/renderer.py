"""
renderer.py

Responsabilidade única: criar frames vazios e desenhar o skeleton
(landmarks + conexões) nesses frames.

Este módulo nunca recebe o frame original de vídeo — só recebe as
coordenadas dos landmarks e desenha em uma "tela em branco".
"""

import numpy as np
import cv2


def create_blank_frame(width, height, color=(0, 0, 0)):
    """Cria um frame vazio (por padrão, preto) com a mesma resolução do vídeo original.

    É esse frame que substitui completamente o frame original — atende
    ao requisito de remover a imagem do atleta, quadra, bola, etc.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if color != (0, 0, 0):
        frame[:] = color
    return frame


def draw_skeleton(frame, landmarks, connections, width, height, visibility_threshold=0.5):
    """Desenha os landmarks e as conexões entre eles no frame informado.

    landmarks: objeto retornado por PoseDetector.process() (coordenadas normalizadas 0-1)
    connections: pares de índices de landmarks que devem ser ligados por uma linha
    visibility_threshold: landmarks com confiança (visibility) abaixo disso não são desenhados
                           -> isto implementa o diferencial "Nível 1" pedido no README
    """
    # As coordenadas do MediaPipe são normalizadas (0.0 a 1.0),
    # então convertemos para pixels usando a largura/altura do frame.
    points = {}
    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility >= visibility_threshold:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points[idx] = (x, y)

    # Desenha as linhas (ossos) apenas quando os dois pontos da conexão
    # foram considerados confiáveis o suficiente
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx in points and end_idx in points:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    # Desenha as articulações (juntas) por cima das linhas
    for point in points.values():
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

    return frame