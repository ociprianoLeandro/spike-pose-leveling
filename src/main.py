import argparse
import csv
import json
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video import open_video, create_video_writer
from pose import init_pose_detector, detect_pose, CONNECTIONS, LANDMARK_NAMES
from renderer import draw_skeleton_on_blank

def save_landmarks(data, output_path):
    """Salva os landmarks exportados em formato CSV ou JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".json"):
        with open(output_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "landmark", "x", "y", "z", "visibility"])
            writer.writeheader()
            writer.writerows(data)
    print(f"Landmarks exportados em: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extração e renderização de Pose com MediaPipe")
    parser.add_argument("--input", type=str, required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", type=str, default="output/video_pose.mp4", help="Caminho do vídeo de saída")
    parser.add_argument("--model", type=str, default="heavy", choices=["lite", "full", "heavy"], help="Variante do modelo MediaPipe (lite, full, heavy)")
    parser.add_argument("--visibility", type=float, default=0.3, help="Limiar de visibilidade para desenhar os landmarks (0.0 a 1.0)")
    parser.add_argument("--export", type=str, default=None, help="Caminho para exportar os dados (ex: output/landmarks.csv)")
    args = parser.parse_args()

    cap, width, height, fps = open_video(args.input)
    detector = init_pose_detector(model_type=args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = create_video_writer(args.output, width, height, fps)

    print(f"Processando {args.input} com modelo '{args.model}'...")

    exported_landmarks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0 and frame_idx > 0:
            timestamp_ms = int((frame_idx / fps) * 1000)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detect_pose(detector, frame_rgb, timestamp_ms)

        if landmarks is not None and args.export:
            for idx, lm in enumerate(landmarks):
                name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"LANDMARK_{idx}"
                exported_landmarks.append({
                    "frame": frame_idx,
                    "landmark": name,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        output_frame = draw_skeleton_on_blank(landmarks, CONNECTIONS, width, height, min_visibility=args.visibility)
        writer.write(output_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    detector.close()
    print(f"Vídeo salvo em: {args.output}")

    if args.export and exported_landmarks:
        save_landmarks(exported_landmarks, args.export)

if __name__ == "__main__":
    main()