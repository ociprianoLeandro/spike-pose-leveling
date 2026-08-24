"""
main.py

Ponto de entrada da aplicação. Liga video.py + pose.py + renderer.py.

Fluxo:
    Vídeo original --(OpenCV)--> frame
                    --(MediaPipe)--> landmarks
                    --(renderer)--> frame vazio com skeleton
                    --(OpenCV)--> vídeo de saída

Uso:
    python src/main.py --input input/ataque.mp4 --output output/ataque_pose.mp4
    python src/main.py --webcam --show
"""

import argparse
import os
import sys
import time

import cv2

# Permite rodar "python src/main.py" a partir da raiz do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video import VideoSource, VideoWriter
from pose import PoseDetector
from renderer import create_blank_frame, draw_skeleton


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrai o skeleton de pose de um vídeo usando MediaPipe Pose."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Caminho para o vídeo de entrada (ex: input/ataque.mp4).",
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Usa a webcam do computador como fonte de vídeo em tempo real.",
    )
    parser.add_argument(
        "--output", type=str, default="output/resultado_pose.mp4",
        help="Caminho do vídeo de saída (usado apenas quando --input é informado).",
    )
    parser.add_argument(
        "--visibility", type=float, default=0.5,
        help="Confiança mínima (visibility, 0-1) para um landmark ser desenhado.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Exibe uma janela com o resultado em tempo real durante o processamento.",
    )
    return parser.parse_args()


def print_landmark_sample(frame_idx, landmarks, landmark_enum):
    """Imprime as coordenadas de alguns landmarks de interesse.

    Isso demonstra, na prática, como acessar programaticamente os dados
    que o MediaPipe extrai — item pedido na seção "Extração dos landmarks".
    """
    interesting = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    print(f"\nFrame {frame_idx}")
    for name in interesting:
        idx = landmark_enum[name].value
        lm = landmarks.landmark[idx]
        print(name)
        print(f"  x: {lm.x:.4f}")
        print(f"  y: {lm.y:.4f}")
        print(f"  z: {lm.z:.4f}")
        print(f"  visibility: {lm.visibility:.4f}")


def run(args):
    # Define a fonte: 0 = webcam padrão, ou o caminho do arquivo
    source = 0 if args.webcam else args.input

    video = VideoSource(source)
    detector = PoseDetector()

    writer = None
    if args.input and not args.webcam:
        # Só faz sentido salvar em disco quando a entrada é um arquivo
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = VideoWriter(args.output, video.width, video.height, video.fps)

    total_frames = 0
    frames_with_detection = 0
    frames_without_detection = 0
    sample_printed = False
    start_time = time.time()

    print("Iniciando processamento... (pressione 'q' na janela para parar, se --show estiver ativo)")

    try:
        while True:
            frame = video.read_frame()
            if frame is None:
                # Fim do vídeo, ou falha ao ler da webcam -> encerra sem quebrar
                break

            total_frames += 1

            # MediaPipe espera frames em RGB; o OpenCV lê em BGR por padrão
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.process(frame_rgb)

            # Este é o frame que vai para o vídeo final -- sempre vazio,
            # nunca uma cópia do frame original
            output_frame = create_blank_frame(video.width, video.height)

            if landmarks is not None:
                frames_with_detection += 1
                draw_skeleton(
                    output_frame,
                    landmarks,
                    detector.connections,
                    video.width,
                    video.height,
                    visibility_threshold=args.visibility,
                )
                if not sample_printed:
                    print_landmark_sample(total_frames, landmarks, detector.landmark_names)
                    sample_printed = True
            else:
                frames_without_detection += 1
                # Nenhum corpo detectado neste frame: seguimos com o frame
                # vazio (sem skeleton) e continuamos o loop normalmente.

            if writer:
                writer.write_frame(output_frame)

            if args.show:
                cv2.imshow("Spike Pose - Skeleton", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário (Ctrl+C).")

    finally:
        elapsed = time.time() - start_time
        video.release()
        if writer:
            writer.release()
        detector.close()
        if args.show:
            cv2.destroyAllWindows()

        print("\n--- Informações de processamento ---")
        print(f"Fonte: {'Webcam' if args.webcam else args.input}")
        print(f"Resolução: {video.width}x{video.height}")
        print(f"FPS: {video.fps:.1f}")
        print(f"Frames processados: {total_frames}")
        print(f"Frames com pose detectada: {frames_with_detection}")
        print(f"Frames sem detecção: {frames_without_detection}")
        if total_frames > 0:
            taxa = (frames_with_detection / total_frames) * 100
            print(f"Taxa de detecção: {taxa:.1f}%")
        print(f"Tempo de processamento: {elapsed:.1f}s")
        if writer:
            print(f"Vídeo salvo em: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    if not args.input and not args.webcam:
        print("Erro: informe --input <caminho_do_video> ou use --webcam.")
        print("Exemplo: python src/main.py --input input/ataque.mp4")
        sys.exit(1)
    run(args)