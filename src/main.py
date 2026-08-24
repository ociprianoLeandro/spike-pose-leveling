import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video import open_video, create_video_writer
from pose import init_pose_detector, detect_pose, CONNECTIONS
from renderer import draw_skeleton_on_blank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", type=str, default="output/video_pose.mp4", help="Caminho do vídeo de saída")
    args = parser.parse_args()

    # Carregando o vídeo e inicializando o detector
    cap, width, height, fps = open_video(args.input)
    detector = init_pose_detector()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = create_video_writer(args.output, width, height, fps)

    print(f"Processando {args.input}...")

    # Loop frame a frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertendo BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extraindo os pontos corporais
        landmarks = detect_pose(detector, frame_rgb)

        # Gerando o frame preto com os pontos desenhados
        output_frame = draw_skeleton_on_blank(landmarks, CONNECTIONS, width, height)

        # Por fim, salva o frame no novo vídeo
        writer.write(output_frame)

    # Apenas liberando os recursos da memória
    cap.release()
    writer.release()
    detector.close()
    print(f"Vídeo salvo em: {args.output}")

if __name__ == "__main__":
    main()