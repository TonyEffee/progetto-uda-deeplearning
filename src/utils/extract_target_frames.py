"""
extract_target_frames.py

Script di preparazione del target dataset per il progetto UDA.

COSA FA:
- Legge, per ogni classe, una cartella "raw" contenente sia foto (jpg/jpeg/png/heic)
  sia video (mov/mp4).
- Copia le foto (convertendo gli HEIC in JPG) direttamente nella cartella di output.
- Estrae frame dai video in modo ADATTIVO: calcola quanti frame servono per
  raggiungere un numero target di immagini per classe, distribuendo l'estrazione
  in modo uniforme su ciascun video e rispettando un intervallo minimo tra i
  frame (per evitare duplicati quasi identici).
- Scrive un manifest.csv che traccia la provenienza di ogni immagine (foto
  originale o frame + nome video sorgente), utile per fare lo split
  train/val SENZA data leakage (frame dello stesso video devono stare
  tutti nello stesso split).

STRUTTURA DI INPUT ATTESA (da creare tu prima di lanciare lo script):

    data/target_raw/
        buildings/   -> foto sciolte (nessun video qui, non serve)
        street/      -> foto sciolte
        sea/         -> foto sciolte
        forest/      -> foto + video .mov/.mp4
        mountain/    -> foto + video .mov/.mp4

STRUTTURA DI OUTPUT (generata dallo script):

    data/target/
        buildings/*.jpg
        street/*.jpg
        sea/*.jpg
        forest/*.jpg          (foto originali + frame estratti)
        mountain/*.jpg
        manifest.csv          (colonna: class, filename, source_type, source_video)

REQUISITI:
    - ffmpeg e ffprobe installati e nel PATH del sistema
      (Mac: brew install ffmpeg)
    - pip install pillow pillow-heif

USO TIPICO (dalla root del repo):

    python -m src.utils.extract_target_frames \\
        --raw-dir data/target_raw \\
        --out-dir data/target \\
        --target-per-class 45 \\
        --min-interval-seconds 3

Puoi anche specificare target diversi per classe, es:

    python -m src.utils.extract_target_frames \\
        --raw-dir data/target_raw \\
        --out-dir data/target \\
        --target-per-class-json '{"forest": 40, "mountain": 45}'
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}


def get_video_duration_seconds(video_path: Path) -> float:
    """Usa ffprobe per ottenere la durata del video in secondi."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, timestamps: list, prefix: str):
    """Estrae frame ai timestamp specificati (in secondi) usando ffmpeg."""
    saved_files = []
    for i, ts in enumerate(timestamps):
        out_file = out_dir / f"{prefix}_frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",  # qualita' alta JPEG
            str(out_file),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        saved_files.append(out_file)
    return saved_files


def convert_heic_to_jpg(src: Path, dst: Path):
    """Converte HEIC/HEIF in JPG usando pillow + pillow-heif."""
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()
    img = Image.open(src).convert("RGB")
    img.save(dst, "JPEG", quality=95)


def process_class(class_name: str, raw_dir: Path, out_dir: Path,
                   target_count: int, min_interval_seconds: float,
                   manifest_rows: list):
    class_raw = raw_dir / class_name
    class_out = out_dir / class_name
    class_out.mkdir(parents=True, exist_ok=True)

    if not class_raw.exists():
        print(f"[WARN] Cartella raw non trovata per la classe '{class_name}': {class_raw}")
        return

    photos = [p for p in class_raw.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    videos = [p for p in class_raw.iterdir() if p.suffix.lower() in VIDEO_EXTS]

    print(f"\n=== Classe '{class_name}' ===")
    print(f"Foto trovate: {len(photos)} | Video trovati: {len(videos)}")

    # 1. Copia/converte le foto
    for photo in photos:
        if photo.suffix.lower() in (".heic", ".heif"):
            dst = class_out / (photo.stem + ".jpg")
            convert_heic_to_jpg(photo, dst)
        else:
            dst = class_out / photo.name
            shutil.copy2(photo, dst)
        manifest_rows.append({
            "class": class_name,
            "filename": dst.name,
            "source_type": "photo",
            "source_video": "",
        })

    # 2. Calcola quanti frame servono in totale e per video
    n_photos = len(photos)
    frames_needed = max(0, target_count - n_photos)

    if frames_needed == 0 or len(videos) == 0:
        print(f"Frame da estrarre: 0 (foto sufficienti o nessun video disponibile)")
        return

    frames_per_video = math.ceil(frames_needed / len(videos))
    print(f"Foto disponibili: {n_photos} | Target: {target_count} "
          f"| Frame da estrarre: {frames_needed} (~{frames_per_video} per video)")

    extracted_total = 0
    for video in videos:
        if extracted_total >= frames_needed:
            break

        try:
            duration = get_video_duration_seconds(video)
        except subprocess.CalledProcessError:
            print(f"[WARN] Impossibile leggere la durata di {video.name}, salto.")
            continue

        # Numero di frame da estrarre da QUESTO video, senza superare il bisogno residuo
        remaining = frames_needed - extracted_total
        n_this_video = min(frames_per_video, remaining)

        # Intervallo minimo tra i frame per evitare duplicati quasi identici
        max_frames_by_interval = max(1, int(duration // min_interval_seconds))
        n_this_video = min(n_this_video, max_frames_by_interval)

        if n_this_video <= 0:
            continue

        # Timestamp distribuiti uniformemente, evitando i primi/ultimi 0.5s
        margin = min(0.5, duration * 0.05)
        usable_duration = max(duration - 2 * margin, 0.1)
        if n_this_video == 1:
            timestamps = [duration / 2]
        else:
            step = usable_duration / (n_this_video - 1)
            timestamps = [margin + i * step for i in range(n_this_video)]

        prefix = video.stem.replace(" ", "_")
        saved_files = extract_frames_ffmpeg(video, class_out, timestamps, prefix)

        for f in saved_files:
            manifest_rows.append({
                "class": class_name,
                "filename": f.name,
                "source_type": "video_frame",
                "source_video": video.name,
            })

        extracted_total += len(saved_files)
        print(f"  - {video.name} ({duration:.1f}s): estratti {len(saved_files)} frame")

    print(f"Totale frame estratti per '{class_name}': {extracted_total}")


def main():
    parser = argparse.ArgumentParser(description="Estrae frame dai video e prepara data/target/")
    parser.add_argument("--raw-dir", type=str, default="data/target_raw",
                         help="Cartella con le sottocartelle per classe (foto+video)")
    parser.add_argument("--out-dir", type=str, default="data/target",
                         help="Cartella di output per il target dataset finale")
    parser.add_argument("--target-per-class", type=int, default=45,
                         help="Numero desiderato di immagini per classe (default per tutte le classi)")
    parser.add_argument("--target-per-class-json", type=str, default=None,
                         help='Override per classi specifiche, es: \'{"forest": 40, "mountain": 45}\'')
    parser.add_argument("--min-interval-seconds", type=float, default=3.0,
                         help="Intervallo minimo (in secondi) tra due frame estratti dallo stesso video")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_overrides = {}
    if args.target_per_class_json:
        per_class_overrides = json.loads(args.target_per_class_json)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Cartella raw non trovata: {raw_dir}\n"
            f"Crea la struttura data/target_raw/<classe>/ con dentro foto e video prima di lanciare lo script."
        )

    class_names = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"Nessuna sottocartella di classe trovata in {raw_dir}")

    print(f"Classi trovate: {class_names}")

    manifest_rows = []
    for class_name in class_names:
        target_count = per_class_overrides.get(class_name, args.target_per_class)
        process_class(class_name, raw_dir, out_dir, target_count,
                       args.min_interval_seconds, manifest_rows)

    # Scrive il manifest
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "filename", "source_type", "source_video"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[SUCCESS] Manifest scritto in {manifest_path}")
    print(f"Totale immagini nel target dataset: {len(manifest_rows)}")


if __name__ == "__main__":
    main()